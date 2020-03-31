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
from NeMO_losses import charbonnierLoss
import NeMO_layers
from keras.models import load_model
from keras.callbacks import (
    ReduceLROnPlateau,
    CSVLogger,
    EarlyStopping,
    ModelCheckpoint,
    TerminateOnNaN)
from NeMO_callbacks import CheckNumericsOps, WeightsSaver

image_size = 80
batch_size = 16
model_name = 'AlexNetSimple'

jsonpath = './utils/CoralClasses.json'
with open(jsonpath) as json_file:
    json_data = json.load(json_file)

# labelkey = json_data["VedConsolidated_ClassDict"]
labelkey = {'Coral':1, 'b':2, 'c':3, 'd':4, 'e':5, 'f':6, 'g':7, 'h':8, 'i':9, 'j':10}
print(labelkey)
num_classes = len(labelkey)

with open("init_args - Kamalika.yml", 'r') as stream:
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
num_channels = 4 # hard-coded for 4 channel

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

#check_num = CheckNumericsOps(validation_data=[np.random.random((1, 224, 224, 3)), 1],
#                             histogram_freq=100)

# log history during model fit
csv_logger = CSVLogger('output/log.csv', append=True, separator=';')

datagen = NeMOImageGenerator(image_shape=[y, x, num_channels],
                                    image_resample=True,
                                    pixelwise_center=True,
                                    pixel_mean=pixel_mean,
                                    pixelwise_std_normalization=True,
                                    augmentation = 0,
                                    channel_shift_range = 0,
                                    random_rotation=True,
                                    pixel_std=pixel_std)
train_generator = datagen.flow_from_NeMOdirectory(train_loader.image_dir,
    source_size=(x,y),
    color_mode='4channel',
    passedclasses = labelkey,
    class_mode = 'categorical',
    batch_size = batch_size,
#    save_to_dir = './Generator_Outputs/',
    shuffle=True)

validation_generator = datagen.flow_from_NeMOdirectory(val_loader.image_dir,
    source_size=(x,y),
    color_mode='4channel',
    passedclasses = labelkey,
    class_mode = 'categorical',
    batch_size = batch_size,
    shuffle=True)

conv_layers = 0
full_layers = 3

conv_params = {"filters": [None],
    "conv_size": [None],
    "conv_strides": [None],
    "padding": ['same'],
    "dilation_rate": [None],
    "pool_size": [None],
    "pool_strides": [None],
    "pad_size": [None],
    "filters_up": [None],
    "upconv_size": [None],
    "upconv_strides": [None],
    "layercombo": [None], 
    "layercombine": [None],           
    "full_filters": [64,64,2], 
    "dropout": [0,0,0.5]}

AlexNetSimple = AlexNetLike(input_shape=(64,64,128), classes=2, weight_decay=3e-3, trainable_encoder=True, weights=None, conv_layers=0, full_layers=full_layers, conv_params=conv_params)
keras.utils.layer_utils.print_summary(AlexNetSimple,line_length=150, positions=[.35, .55, .65, 1.])

optimizer = keras.optimizers.Adam(1e-4)
AlexNetSimple.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'], sample_weight_mode='temporal')

print("Memory required (GB): ", get_model_memory_usage(batch_size, AlexNetSimple))

# AlexNetSimple.fit_generator(train_generator,
#     steps_per_epoch=100,
#     epochs=100,
#     validation_data=validation_generator,
#     validation_steps=20,
#     verbose=1,
#     callbacks=[lr_reducer, early_stopper, nan_terminator, checkpointer])



