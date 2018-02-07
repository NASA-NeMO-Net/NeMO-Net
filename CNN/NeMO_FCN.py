import os
import yaml
import datetime
import numpy as np
import keras
import keras.backend as K
import tensorflow as tf
import sys
sys.path.append("./utils/") # Adds higher directory to python modules path.
#sys.path.append("./tmp/")
from NeMO_models import FCN
from NeMO_generator import NeMOImageGenerator, ImageSetLoader
from keras.callbacks import (
    ReduceLROnPlateau,
    CSVLogger,
    EarlyStopping,
    ModelCheckpoint,
    TerminateOnNaN)
from NeMO_callbacks import CheckNumericsOps, WeightsSaver


os.environ["CUDA_VISIBLE_DEVICES"] = "0"

global _SESSION
config = tf.ConfigProto(allow_soft_placement=True)
config.gpu_options.allow_growth = True
_SESSION = tf.Session(config=config)
K.set_session(_SESSION)

image_size = 150
batch_size = 120

imgpath = '../Images/BTPB-WV2-2012-15-8Band-mosaic-GeoTiff-Sample-AOI/BTPB-WV2-2012-15-8Band-mosaic-GeoTiff-Sample-AOI.tif'
tfwpath = '../Images/BTPB-WV2-2012-15-8Band-mosaic-GeoTiff-Sample-AOI/BTPB-WV2-2012-15-8Band-mosaic-GeoTiff-Sample-AOI.tfw'
truthpath = '../Images/BIOT-PerosBanhos-sample-habitat-map/BIOT-PerosBanhos-sample-habitat-map.shp'
PerosBanhos = coralutils.CoralData(imgpath, Truthpath=truthpath, load_type="raster", tfwpath=tfwpath)
labelkey = PerosBanhos.class_labels

with open("init_args - AlexNetParallel_Raster.yml", 'r') as stream:
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

checkpointer = ModelCheckpoint(filepath="./tmp/fcn_vgg16_weights.h5", verbose=1, save_best_only=True)
lr_reducer = ReduceLROnPlateau(monitor='val_loss',
                               factor=np.sqrt(0.1),
                               cooldown=0,
                               patience=10, min_lr=1e-12)
early_stopper = EarlyStopping(monitor='val_loss',
                              min_delta=0.001,
                              patience=30)
nan_terminator = TerminateOnNaN()
SaveWeights = WeightsSaver(filepath='./weights/', N=10)
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
    target_size=(x,y),
    color_mode=train_loader.color_mode,
    classes = labelkey,
    class_mode = 'categorical',
    batch_size = batch_size,
    shuffle=True)

validation_generator = datagen.flow_from_NeMOdirectory(val_loader.image_dir,
    target_size=(x,y),
    color_mode=val_loader.color_mode,
    classes = labelkey,
    class_mode = 'categorical',
    batch_size = batch_size,
    shuffle=True)

num_classes = train_generator.num_class

fcn_vgg16 = FCN(input_shape=(y, x, num_channels), classes=num_classes, weight_decay=3e-3,
                weights='imagenet', trainable_encoder=True)
optimizer = keras.optimizers.Adam(1e-4)

fcn_vgg16.compile(optimizer=optimizer,
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

fcn_vgg16.fit_generator(
    datagen.flow_from_imageset(
        class_mode='categorical',
        classes=4,
        batch_size=20,
        shuffle=True,
        image_set_loader=train_loader),
    steps_per_epoch=50,
    epochs=100,
        batch_size=10,
        shuffle=True,
        image_set_loader=train_loader),
    steps_per_epoch=80,
    epochs=2,
    validation_data=datagen.flow_from_imageset(
        class_mode='categorical',
        classes=4,
        batch_size=20,
        shuffle=True,
        image_set_loader=val_loader),
    validation_steps=5,
    verbose=1,
    callbacks=[lr_reducer, early_stopper, nan_terminator, checkpointer])
    callbacks=[lr_reducer, early_stopper, nan_terminator,checkpointer, csv_logger, SaveWeights])

fcn_vgg16.save('./tmp/fcn_vgg16_model.h5')
