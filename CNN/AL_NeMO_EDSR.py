import os
import yaml
import datetime
import numpy as np
import json
import keras
import keras.backend as K
from keras.losses import mean_absolute_error, mean_squared_error
from keras.models import Model, Sequential, load_model
import tensorflow as tf
import sys
sys.path.append("./utils/") # Adds higher directory to python modules path.
import NeMO_layers

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
from keras.losses import mean_absolute_error, mean_squared_error
from keras.callbacks import (
    ReduceLROnPlateau,
    CSVLogger,
    EarlyStopping,
    ModelCheckpoint,
    TerminateOnNaN)
from NeMO_callbacks import CheckNumericsOps, WeightsSaver
from NeMO_losses import charbonnierLoss

image_size = 128
batch_size = 16
mag = 2
model_name = 'SRx2_Fiji_4channel_EDSR'

jsonpath = './utils/CoralClasses.json'
with open(jsonpath) as json_file:
    json_data = json.load(json_file)

labelkey = json_data["Fiji_ClassDict"]
num_classes = len(labelkey)

with open("init_args - SRx2_Fiji.yml", 'r') as stream:
    try:
        init_args = yaml.load(stream)
    except yaml.YAMLError as exc:
        print(exc)

train_loader = ImageSetLoader(**init_args['image_set_loader']['train'])
val_loader = ImageSetLoader(**init_args['image_set_loader']['val'])

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

# log history during model fit
csv_logger = CSVLogger('output/log.csv', append=True, separator=';')

datagen = NeMOImageGenerator(image_shape=[y, x, num_channels],
                                    image_resample=True,
                                    pixelwise_center=True,
                                    pixel_mean=pixel_mean,
                                    pixelwise_std_normalization=True,
                                    random_rotation=True,
                                    pixel_std=pixel_std,
                                    image_or_label="image")
train_generator = datagen.flow_from_NeMOdirectory(train_loader.image_dir,
    FCN_directory=train_loader.label_dir,
    source_size=(x,y),
    target_size=(x*mag,y*mag),
    color_mode="4channel_delete",
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
    color_mode="4channel_delete",
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

# scaling layer is default = 0.1, need to update functions later to accept incoming values
RCU = ("cacs","")

# Upsampling done by pixel_shuffle, which only does x2 internally
conv_params = {"filters":[64,64,[256,num_channels]],
    "conv_size": [(3,3), (3,3), (3,3)],
    "conv_strides": [(1,1), (1,1), (1,1)],
    "padding": ['same','same','same'],
    "dilation_rate": [(1,1), (1,1), (1,1)],
    "scaling": [0.1,0.1,0.1],
    "filters_up": [None, None, None],
    "upconv_size": [None, None, None],
    "upconv_strides": [None, None, None],
    "upconv_type": ["","","pixel_shuffle"],
    "layercombo": ["c", ([RCU]*16 + ["c"],""), "cuc"],
    "layercombine": ["sum","sum","sum"],
    "full_filters": [1024,1024],
    "dropout": [0,0]}


SR_EDSR = AlexNetLike(input_shape=(y, x, num_channels), classes=num_classes, weight_decay=3e-3, trainable_encoder=True, weights=None, conv_layers=conv_layers, full_layers=0, conv_params=conv_params, onebyoneconv=False, reshape=False)
# SR_EDSR = load_model('./tmp/SRx2_Fiji_4channel_EDSR.h5', custom_objects={'PixelShuffler':NeMO_layers.PixelShuffler, 'BilinearUpSampling2D':NeMO_layers.BilinearUpSampling2D, 'charbonnierLoss': charbonnierLoss})

optimizer = keras.optimizers.Adam(1e-4)

keras.utils.layer_utils.print_summary(SR_EDSR, line_length=150, positions=[.35, .55, .65, 1.])
# TestArchitecture.summary()
# TestArchitecture.compile(loss=charbonnierLoss, optimizer=optimizer, metrics=['accuracy'], sample_weight_mode='temporal')
SR_EDSR.compile(loss=mean_absolute_error, optimizer=optimizer)
print("Memory required (GB): ", get_model_memory_usage(batch_size, SR_EDSR))

SR_EDSR.fit_generator(train_generator,
    steps_per_epoch=200,
    epochs=100,
    validation_data=validation_generator,
    validation_steps=20,
    verbose=1,
    callbacks=[lr_reducer, early_stopper, nan_terminator, checkpointer])