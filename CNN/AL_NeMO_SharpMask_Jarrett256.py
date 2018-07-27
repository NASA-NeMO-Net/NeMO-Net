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
from NeMO_models import AlexNetLike, SharpMask_FCN
from NeMO_generator import NeMOImageGenerator, ImageSetLoader
from NeMO_backend import get_model_memory_usage
import NeMO_layers
from keras.models import load_model
from keras.callbacks import (
    ReduceLROnPlateau,
    CSVLogger,
    EarlyStopping,
    ModelCheckpoint,
    TerminateOnNaN)
from NeMO_callbacks import CheckNumericsOps, WeightsSaver

image_size = 256
batch_size = 16
model_name = 'SharpMask_Jarrett256_v2'

jsonpath = './utils/CoralClasses.json'
with open(jsonpath) as json_file:
    json_data = json.load(json_file)

labelkey = json_data["VedConsolidated_ClassDict"]
num_classes = len(labelkey)

with open("init_args - Jarrett.yml", 'r') as stream:
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
pixel_mean =0*np.ones(num_channels)
pixel_std = 1*np.ones(num_channels)
# channel_shift_range = [0.01]*num_channels
# rescale = np.asarray([[0.95,1.05]]*num_channels)

checkpointer = ModelCheckpoint(filepath="./tmp/" + model_name + ".h5", verbose=1, monitor='val_acc', mode='max', save_best_only=True)
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

#check_num = CheckNumericsOps(validation_data=[np.random.random((1, 224, 224, 3)), 1],
#                             histogram_freq=100)

# log history during model fit
csv_logger = CSVLogger('output/log.csv', append=True, separator=';')

datagen = NeMOImageGenerator(image_shape=[y, x, num_channels],
                                    image_resample=True,
                                    pixelwise_center=True,
                                    pixel_mean=pixel_mean,
                                    pixelwise_std_normalization=True,
                                    channel_shift_range = 0.1,
                                    random_rotation=True,
                                    pixel_std=pixel_std)
train_generator = datagen.flow_from_NeMOdirectory(train_loader.image_dir,
    FCN_directory=train_loader.label_dir,
    source_size=(x,y),
    target_size=(x,y),
    color_mode=train_loader.color_mode,
    passedclasses = labelkey,
    class_mode = 'categorical',
    batch_size = batch_size,
    shuffle=True)

validation_generator = datagen.flow_from_NeMOdirectory(val_loader.image_dir,
    FCN_directory=val_loader.label_dir,
    source_size=(x,y),
    target_size=(x,y),
    color_mode=val_loader.color_mode,
    passedclasses = labelkey,
    class_mode = 'categorical',
    batch_size = batch_size,
    shuffle=True)

conv_layers = 5
full_layers = 0

# First 4 megablocks of the resnet-50 architecture
conv_params = {"filters": [[64] , [64,64,256]*3, [128,128,512]*3, [256,256,1024]*3, [512]],
    "conv_size": [[(7,7)] , [(1,1),(3,3),(1,1)]*3, [(1,1),(3,3),(1,1)]*3, [(1,1),(3,3),(1,1)]*3, [(1,1)]],
    "conv_strides": [(1,1), [(2,2)]+[(1,1)]*8 , [(2,2)]+[(1,1)]*8 , [(2,2)]+[(1,1)]*8, [(1,1)]],
    "padding": ['same', 'same', 'same', 'same', 'same'],
    "dilation_rate": [(1,1), (1,1), (1,1), (1,1), (1,1)],
    "pool_size": [(3,3), (1,1), (1,1), (1,1), (1,1)],
    "pool_strides": [(2,2), (1,1), (1,1), (1,1), (1,1)],
    "pad_size": [(0,0), (0,0), (0,0), (0,0), (0,0)],
    "filters_up": [None]*conv_layers,
    "upconv_size": [None]*conv_layers,
    "upconv_strides": [None]*conv_layers,
    "layercombo": ["cbap", "cbacbacs"+"bacbacbacs"*2, "bacbacbacs"*3, "bacbacbacs"*3, "c"],
    "full_filters": [1024,1024],
    "dropout": [0,0]}

bridge_params = {"filters": [None, [128,128,64], [64,64,32], [32,32,16], [16,16,8]],
    "conv_size": [None, (3,3),(3,3),(3,3), (3,3)],
    "filters_up": [None]*5,
    "upconv_size": [None]*5,
    "upconv_strides": [None]*5,
    "layercombo": ["", "cbacbac", "cbacbac", "cbacbac", "cbacbac"]}

prev_params = {"filters": [None, [128,64], [64,32], [32,16], [16,8]],
    "conv_size": [None, (3,3),(3,3),(3,3), (3,3)],
    "filters_up": [None]*5,
    "upconv_size": [None]*5,
    "upconv_strides": [None]*5,
    "layercombo": ["", "cbac", "cbac", "cbac", "cbac"]} 

next_params = {"filters": [None, 128,64,32,16],
    "conv_size": [None, (3,3),(3,3),(3,3), (3,3)],
    "filters_up": [None]*5,
    "upconv_size": [None]*5,
    "upconv_strides": [None]*5,
    "layercombo": ["", "ba", "ba", "ba", "ba"]} 

decoder_index = [0,1,2,3,4]
upsample = [False,True,True,True,True]
scales= [1,1,1,1,1]

# SharpMask = SharpMask_FCN(input_shape=(y,x,num_channels), classes=num_classes, decoder_index = decoder_index, weight_decay=3e-3, trainable_encoder=True, weights=None,
#     conv_layers=conv_layers, full_layers=full_layers, conv_params=conv_params, scales=scales, 
#     bridge_params=bridge_params, prev_params=prev_params, next_params=next_params, upsample=upsample)

SharpMask = load_model('./tmp/SharpMask_Jarrett256_v2.h5', custom_objects={'BilinearUpSampling2D':NeMO_layers.BilinearUpSampling2D})

optimizer = keras.optimizers.Adam(1e-4)

SharpMask.summary()
SharpMask.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'], sample_weight_mode='temporal')

print("Memory required (GB): ", get_model_memory_usage(batch_size, SharpMask))

SharpMask.fit_generator(train_generator,
    steps_per_epoch=50,
    epochs=100,
    validation_data=validation_generator,
    validation_steps=5,
    verbose=1,
    callbacks=[lr_reducer, early_stopper, nan_terminator, checkpointer])