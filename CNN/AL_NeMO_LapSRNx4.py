import os
import yaml
import datetime
import numpy as np
import json
import keras
import keras.backend as K
import tensorflow as tf

os.environ["CUDA_VISIBLE_DEVICES"] = "2"

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

image_size = 64
batch_size = 12
mag = 4
model_name = 'SRx4_Fiji_4channel'

jsonpath = './utils/CoralClasses.json'
with open(jsonpath) as json_file:
    json_data = json.load(json_file)

labelkey = json_data["Fiji_ClassDict"]
# labelkey = {'Sand': 0, 'Branching': 1, 'Mounding': 2, 'Rock':3}
num_classes = len(labelkey)

with open("init_args - SRx4_Fiji.yml", 'r') as stream:
    try:
        init_args = yaml.load(stream)
    except yaml.YAMLError as exc:
        print(exc)

train_loader = ImageSetLoader(**init_args['image_set_loader']['train'])
val_loader = ImageSetLoader(**init_args['image_set_loader']['val'])

# if train_loader.color_mode == 'rgb':
#     num_channels = 3
# elif train_loader.color_mode == '8channel':
#     num_channels = 8

num_channels = 4 # WV2 to Sentinel hard-code

y = train_loader.target_size[1]
x = train_loader.target_size[0]
pixel_mean =100*np.ones(num_channels)
pixel_std = 100*np.ones(num_channels)
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
    target_size=(x*mag,y*mag),
    color_mode="4channel",
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
    target_size=(x*mag,y*mag),
    color_mode="4channel",
    passedclasses = labelkey,
    class_mode = 'categorical',
    batch_size = batch_size,
    shuffle=True,
    image_or_label="image")
    # save_to_dir='./tmpbatchsave/',
    # save_format='png',
    # image_or_label="image")


conv_layers = 3
full_layers = 0

conv_params = {"filters":[64,64,64],
    "conv_size": [[([(1,1),(3,3)], [(1,1),(6,6)], [(3,3),(1,1)]), (3,3)], (3,3), (3,3)],
    "conv_strides": [(1,1), (1,1), (1,1)],
    "padding": ['same','same', 'same'],
    "dilation_rate": [(1,1), (1,1), (1,1)],
    "filters_up": [None, 64, 64],
    "upconv_size": [None, (3,3), (3,3)],
    "upconv_strides": [None, (2,2), (2,2)],
    "upconv_type": ["","2dtranspose","2dtranspose"],
    "layercombo": [[("cbacba","cbacba","cbacba"),"c"], [([("cbacbacbacba",""), "cbacbacbacba"], ""), "uba"], [([("cbacbacbacba",""), "cbacbacbacba"], ""), "uba"]],
    "layercombine": ["cat",["sum","sum"], ["sum","sum"]],
    "full_filters": [1024,1024],
    "dropout": [0,0]}

bridge_params = {"filters": [None,num_channels,num_channels],
    "conv_size": [None,(3,3),(3,3)],
    "conv_strides": [None,(1,1),(1,1)],
    "padding": ['same','same','same'],
    "dilation_rate": [None, (1,1), (1,1)],
    "layercombo": ["", "ca", "ca"]}

prev_params = {"filters_up": [None,num_channels,num_channels],
    "upconv_size": [None, (3,3), (3,3)],
    "upconv_strides": [None, (2,2), (2,2)],
    "upconv_type": ["","2dtranspose", "2dtranspose"],
    "layercombo": ["", "u", "u"]} 

next_params = {"layercombo": ["", "", ""]} 

# conv_params = {"filters":[([64,128],[64,128],[64,128]), 128, ([128],[]), ([128],[]), ([128],[]), ([128],[]),[], ([128],[]), ([128],[]), ([128],[]), ([128],[]),[]],
#     "conv_size": [([(1,1),(3,3)],[(1,1),(6,6)],[(3,3),(1,1)]), (3,3), ((3,3),[]), ((3,3),[]), ((3,3),[]), ((3,3),[]), [], ((3,3),[]), ((3,3),[]), ((3,3),[]), ((3,3),[]), []],
#     "conv_strides": [(1,1), (1,1), (1,1), (1,1), (1,1), (1,1), [], (1,1), (1,1), (1,1), (1,1), []],
#     "padding": ['same','same','same','same','same','same','same', 'same','same','same','same','same'],
#     "dilation_rate": [(1,1), (1,1), (1,1), (1,1), (1,1), (1,1), [], (1,1), (1,1), (1,1), (1,1), []],
#     "pool_size": [(2,2), (2,2), (2,2), (2,2), (2,2), (2,2),[], (2,2), (2,2), (2,2), (2,2),[]],
#     "pool_strides": [(2,2), (2,2), (2,2), (2,2), (2,2), (2,2),[], (2,2), (2,2), (2,2), (2,2),[]],
#     "pad_size": [(0,0), (0,0), (0,0), (0,0), (0,0), (0,0), [], (0,0), (0,0), (0,0), (0,0), []],
#     "filters_up": [[], [], [], [], [], [], 3, [], [], [], [], 3],
#     "upconv_size": [[], [], [], [], [], [], (3,3), [], [], [], [], (3,3)],
#     "upconv_strides": [[], [], [], [], [], [], (2,2), [], [], [], [], (2,2)],
#     "layercombo": [("cbacba","cbacba","cbacba"), "c", ("cba",""), ("cba",""), ("cba",""), ("cba",""), "uba", ("cba",""), ("cba",""), ("cba",""), ("cba",""), "uba"],
#     "full_filters": [1024],
#     "dropout": [0,0]}

# bridge_params = {"filters": [3],
#     "conv_size": [(2,2)],
#     "layercombo": ["", "ca", "ca"]}

# prev_params = { "filters_up": [3],
#     "upconv_size": [(2,2)],
#     "upconv_strides": [(2,2)],
#     "layercombo": ["", "u", "u"]} 

# next_params = {"layercombo": ["", "", "",""]} 

decoder_index = [1,0,2]     # Input is added manually in the model, last one is not used
scales= [1,1,1]

# SharpMask = SharpMask_FCN(input_shape=(y,x,num_channels), classes=num_classes, decoder_index = decoder_index, weight_decay=3e-3, trainable_encoder=True, weights=None,
#     conv_layers=conv_layers, full_layers=full_layers, conv_params=conv_params, scales=scales, 
#     bridge_params=bridge_params, prev_params=prev_params, next_params=next_params, upsample=upsample)

TestArchitecture = TestModel(input_shape=(y, x, num_channels), classes=num_classes, decoder_index=decoder_index, weight_decay=3e-3, trainable_encoder=True, weights=None,
    conv_layers=conv_layers, full_layers=0, conv_params=conv_params, scales=scales, bridge_params=bridge_params, prev_params=prev_params, next_params=next_params)

optimizer = keras.optimizers.Adam(1e-4)

TestArchitecture.summary()
# TestArchitecture.compile(loss=charbonnierLoss, optimizer=optimizer, metrics=['accuracy'], sample_weight_mode='temporal')
TestArchitecture.compile(loss=charbonnierLoss, optimizer=optimizer)
print("Memory required (GB): ", get_model_memory_usage(batch_size, TestArchitecture))

TestArchitecture.fit_generator(train_generator,
    steps_per_epoch=400,
    epochs=100,
    validation_data=validation_generator,
    validation_steps=20,
    verbose=1,
    callbacks=[lr_reducer, early_stopper, nan_terminator, checkpointer])