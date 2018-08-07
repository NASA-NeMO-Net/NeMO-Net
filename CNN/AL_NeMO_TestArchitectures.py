import os
import yaml
import datetime
import numpy as np
import json
import keras
import keras.backend as K
import tensorflow as tf

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

global _SESSION
config = tf.ConfigProto(allow_soft_placement=True)
config.gpu_options.allow_growth = True
_SESSION = tf.Session(config=config)
K.set_session(_SESSION)

import sys
sys.path.append("./utils/") # Adds higher directory to python modules path.
import loadcoraldata_utils as coralutils
from NeMO_models import AlexNetLike, SharpMask_FCN, TestModel
from NeMO_generator import NeMOImageGenerator, ImageSetLoader
from NeMO_backend import get_model_memory_usage
from keras.callbacks import (
    ReduceLROnPlateau,
    CSVLogger,
    EarlyStopping,
    ModelCheckpoint,
    TerminateOnNaN)
from NeMO_callbacks import CheckNumericsOps, WeightsSaver
from NeMO_losses import charbonnierLoss

image_size = 32
batch_size = 64
model_name = 'SRx2'

jsonpath = './utils/CoralClasses.json'
with open(jsonpath) as json_file:
    json_data = json.load(json_file)

#labelkey = json_data["VedConsolidated_ClassDict"]
labelkey = {'Sand': 0, 'Branching': 1, 'Mounding': 2, 'Rock':3}
num_classes = len(labelkey)

with open("init_args - SR.yml", 'r') as stream:
    try:
        init_args = yaml.load(stream)
    except yaml.YAMLError as exc:
        print(exc)

train_loader = ImageSetLoader(**init_args['image_set_loader']['train'])
val_loader = ImageSetLoader(**init_args['image_set_loader']['val'])

if train_loader.color_mode == 'rgb':
    num_channels = 3
elif train_loader.color_mode == '8channel':
    num_channels = 8

y = train_loader.target_size[1]
x = train_loader.target_size[0]
pixel_mean =127.5*np.ones(num_channels)
pixel_std = 127.5*np.ones(num_channels)
# channel_shift_range = [0.01]*num_channels
# rescale = np.asarray([[0.95,1.05]]*num_channels)

checkpointer = ModelCheckpoint(filepath="./tmp/" + model_name + ".h5", verbose=1, monitor='val_loss', mode='min', save_best_only=True)
lr_reducer = ReduceLROnPlateau(monitor='val_loss',
                               factor=np.sqrt(0.1),
                               cooldown=0,
                               patience=10, min_lr=1e-12)
early_stopper = EarlyStopping(monitor='val_loss',
                              min_delta=0.001,
                              patience=30)
nan_terminator = TerminateOnNaN()
SaveWeights = WeightsSaver(filepath='./weights/', model_name=model_name, N=10)
#csv_logger = CSVLogger('output/tmp_fcn_vgg16.csv')
    #'output/{}_fcn_vgg16.csv'.format(datetime.datetime.now().isoformat()))

# log history during model fit
csv_logger = CSVLogger('output/log.csv', append=True, separator=';')

datagen = NeMOImageGenerator(image_shape=[y, x, num_channels],
                                    image_resample=True,
                                    pixelwise_center=True,
                                    pixel_mean=pixel_mean,
                                    pixelwise_std_normalization=True,
                                    random_rotation=False,
                                    pixel_std=pixel_std,
                                    image_or_label="image")
train_generator = datagen.flow_from_NeMOdirectory(train_loader.image_dir,
    FCN_directory=train_loader.label_dir,
    source_size=(x,y),
    target_size=(x*2,y*2),
    color_mode=train_loader.color_mode,
    passedclasses = labelkey,
    class_mode = 'categorical',
    batch_size = batch_size,
    shuffle=True,
    image_or_label="image")
    # save_to_dir='./tmpbatchsave/',
    # save_format='png',
    # image_or_label="image")

validation_generator = datagen.flow_from_NeMOdirectory(val_loader.image_dir,
    FCN_directory=val_loader.label_dir,
    source_size=(x,y),
    target_size=(x*2,y*2),
    color_mode=val_loader.color_mode,
    passedclasses = labelkey,
    class_mode = 'categorical',
    batch_size = batch_size,
    shuffle=True,
    image_or_label="image")
    # save_to_dir='./tmpbatchsave/',
    # save_format='png',
    # image_or_label="image")

conv_layers = 7
full_layers = 0

conv_params = {"filters":[([64,128],[64,128],[64,128]), 128, ([128],[]), ([128],[]), ([128],[]), ([128],[]),[]],
    "conv_size": [([(1,1),(3,3)],[(1,1),(6,6)],[(3,3),(1,1)]), (3,3), ((3,3),[]), ((3,3),[]), ((3,3),[]), ((3,3),[]), []],
    "conv_strides": [(1,1), (1,1), (1,1), (1,1), (1,1), (1,1), []],
    "padding": ['same','same','same','same','same','same','same'],
    "dilation_rate": [(1,1), (1,1), (1,1), (1,1), (1,1), (1,1), []],
    "pool_size": [(2,2), (2,2), (2,2), (2,2), (2,2), (2,2),[]],
    "pool_strides": [(2,2), (2,2), (2,2), (2,2), (2,2), (2,2),[]],
    "pad_size": [(0,0), (0,0), (0,0), (0,0), (0,0), (0,0), []],
    "filters_up": [[], [], [], [], [], [], 3],
    "upconv_size": [[], [], [], [], [], [], (3,3)],
    "upconv_strides": [[], [], [], [], [], [], (2,2)],
    "layercombo": [("cbacba","cbacba","cbacba"), "c", ("cba",""), ("cba",""), ("cba",""), ("cba",""), "uba"],
    "full_filters": [1024,1024],
    "dropout": [0,0]}

# conv_params = {"filters": [[64, 128], [128,[64,128],128]],
#     "conv_size": [[(7,7), (5,5)], [(1,1),[(2,2),(3,3)],(5,5)]],
#     "conv_strides": [(1,1),(1,1)],
#     "padding": ['same', 'same'],
#     "dilation_rate": [(1,1), (1,1)],
#     "pool_size": [(2,2), (2,2)],
#     "pool_strides": [(2,2), (2,2)],
#     "pad_size": [(0,0), (0,0)],
#     "layercombo": ["cbacp", ["cba","cacba","cba"]],
#     "full_filters": [1024,1024],
#     "dropout": [0,0]}

bridge_params = {"filters": [None,3],
    "conv_size": [None,(2,2)],
    "filters_up": [None,None],
    "upconv_size": [None,None],
    "upconv_strides": [None,None],
    "layercombo": ["", "ca"]}

prev_params = {"filters": [None, None],
    "conv_size": [None, None],
    "filters_up": [None,3],
    "upconv_size": [None, (2,2)],
    "upconv_strides": [None, (2,2)],
    "layercombo": [None, "u"]} 

next_params = {"filters": [None,None],
    "conv_size": [None, None],
    "filters_up": [None, None],
    "upconv_size": [None, None],
    "upconv_strides": [None, None],
    "layercombo": ["", ""]} 

decoder_index = [0]     # Input is added manually in the model
upsample = [False,False]
scales= [1,1]

# SharpMask = SharpMask_FCN(input_shape=(y,x,num_channels), classes=num_classes, decoder_index = decoder_index, weight_decay=3e-3, trainable_encoder=True, weights=None,
#     conv_layers=conv_layers, full_layers=full_layers, conv_params=conv_params, scales=scales, 
#     bridge_params=bridge_params, prev_params=prev_params, next_params=next_params, upsample=upsample)

TestArchitecture = TestModel(input_shape=(y, x, num_channels), classes=num_classes, decoder_index=decoder_index, weight_decay=3e-3, trainable_encoder=True, weights=None,
    conv_layers=conv_layers, full_layers=0, conv_params=conv_params, scales=scales, bridge_params=bridge_params, prev_params=prev_params, next_params=next_params, upsample=upsample)

optimizer = keras.optimizers.Adam(1e-4)

TestArchitecture.summary()
# TestArchitecture.compile(loss=charbonnierLoss, optimizer=optimizer, metrics=['accuracy'], sample_weight_mode='temporal')
TestArchitecture.compile(loss=charbonnierLoss, optimizer=optimizer)
print("Memory required (GB): ", get_model_memory_usage(batch_size, TestArchitecture))

TestArchitecture.fit_generator(train_generator,
    steps_per_epoch=125,
    epochs=100,
    validation_data=validation_generator,
    validation_steps=20,
    verbose=1,
    callbacks=[lr_reducer, early_stopper, nan_terminator, checkpointer])