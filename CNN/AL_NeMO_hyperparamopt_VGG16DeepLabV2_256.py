import os
import yaml
import datetime
import numpy as np
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
from NeMO_models import VGG16_DeepLabV2
from NeMO_generator import NeMOImageGenerator, ImageSetLoader
from NeMO_backend import get_model_memory_usage
from keras.callbacks import (
    ReduceLROnPlateau,
    CSVLogger,
    EarlyStopping,
    ModelCheckpoint,
    TerminateOnNaN)
from NeMO_callbacks import CheckNumericsOps, WeightsSaver

image_size = 256
batch_size = 6
model_name = 'VGG16DeepLab_Raster256'

imgpath = '../Images/BTPB-WV2-2012-15-8Band-mosaic-GeoTiff-Sample-AOI/BTPB-WV2-2012-15-8Band-mosaic-GeoTiff-Sample-AOI.tif'
tfwpath = '../Images/BTPB-WV2-2012-15-8Band-mosaic-GeoTiff-Sample-AOI/BTPB-WV2-2012-15-8Band-mosaic-GeoTiff-Sample-AOI.tfw'
truthpath = '../Images/BIOT-PerosBanhos-sample-habitat-map/BIOT-PerosBanhos-sample-habitat-map.shp'
PerosBanhos = coralutils.CoralData(imgpath, Truthpath=truthpath, load_type="raster", tfwpath=tfwpath)
PerosBanhos.load_PB_consolidated_classes()

#labelkey = PerosBanhos.class_labels
labelkey = PerosBanhos.consol_labels
num_classes = len(PerosBanhos.PB_consolidated_classes)

with open("init_args - VGG16DeepLab_Raster256.yml", 'r') as stream:
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
pixel_mean =1023.5*np.ones(num_channels)
pixel_std = 1023.5*np.ones(num_channels)

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
                                    pixel_std=pixel_std)
train_generator = datagen.flow_from_NeMOdirectory(train_loader.image_dir,
    FCN_directory=train_loader.label_dir,
    target_size=(x,y),
    color_mode=train_loader.color_mode,
    classes = labelkey,
    class_weights = PerosBanhos.consolclass_weights,
    class_mode = 'categorical',
    batch_size = batch_size,
    shuffle=True)

validation_generator = datagen.flow_from_NeMOdirectory(val_loader.image_dir,
    FCN_directory=val_loader.label_dir,
    target_size=(x,y),
    color_mode=val_loader.color_mode,
    classes = labelkey,
    class_weights = PerosBanhos.consolclass_weights,
    class_mode = 'categorical',
    batch_size = batch_size,
    shuffle=True)

conv_layers = 5
full_layers = 0

conv_params = {"filters": [64,128,256,512,512],
    "conv_size": [(3,3),(3,3),(3,3),(3,3),(3,3)],
    "conv_strides": [(1,1),(1,1),(1,1),(1,1),(1,1)],
    "padding": ['same','same','same','same','same'],
    "dilation_rate": [(1,1),(1,1),(1,1),(1,1),(2,2)],
    "pool_size": [(2,2),(2,2),(2,2),(3,3),(3,3)],
    "pool_strides": [(2,2),(2,2),(2,2),(1,1),(1,1)],
    "pad_size": [(0,0),(0,0),(1,1),(1,1),(1,1)],
    "layercombo": ["cacapb","cacapb","cacacapb","cacacazpb","cacacazp"],
    "full_filters": [1024,1024],
    "dropout": [0,0]}

parallelconv_params = {"filters": [[1024,1024,num_classes]],
    "conv_size": [[(3,3),(1,1),(1,1)]],
    "conv_strides":  [[(1,1),(1,1),(1,1)]],
    "padding": ['valid','valid','valid','valid'],
    "dilation_rate": [[(6,6),(1,1),(1,1)], [(12,12),(1,1),(1,1)], [(18,18),(1,1),(1,1)], [(24,24),(1,1),(1,1)]],
    "pool_size": [[(2,2),(2,2),(2,2)]], #doesn't matter
    "pool_strides": [[(2,2),(2,2),(2,2)]], #doesn't matter
    "pad_size": [[(6,6),(0,0),(0,0)], [(12,12),(0,0),(0,0)], [(18,18),(0,0),(0,0)], [(24,24),(0,0),(0,0)]],
    "layercombo": ["zcadcadc","zcadcadc","zcadcadc","zcadcadc"],
    "full_filters": [4096,2048],
    "dropout": [0.5,0.5]}

VGG16_DeepLab = VGG16_DeepLabV2(input_shape=(y, x, num_channels), classes=num_classes, weight_decay=3e-3, batch_size=batch_size, 
                weights=None, trainable_encoder=True, conv_layers=5, full_layers=0, conv_params=conv_params, parallel_layers=4, parallelconv_params=parallelconv_params)
optimizer = keras.optimizers.Adam(1e-4)

VGG16_DeepLab.summary()
VGG16_DeepLab.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'], sample_weight_mode='temporal')

# print("Memory required (GB): ", get_model_memory_usage(batch_size, VGG16_DeepLab))

VGG16_DeepLab.fit_generator(train_generator,
    steps_per_epoch=100,
    epochs=100,
    validation_data=validation_generator,
    validation_steps=100,
    verbose=1,
    callbacks=[lr_reducer, early_stopper, nan_terminator, checkpointer])