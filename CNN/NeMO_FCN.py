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

with open("init_args.yml", 'r') as stream:
    try:
        init_args = yaml.load(stream)
    except yaml.YAMLError as exc:
        print(exc)

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

datagen = NeMOImageGenerator(image_shape=[image_size, image_size, 3],
                                    image_resample=True,
                                    pixelwise_center=True,
                                    pixel_mean=[127.5, 127.5, 127.5],
                                    pixelwise_std_normalization=True,
                                    pixel_std=[127.5, 127.5, 127.5])

train_loader = ImageSetLoader(**init_args['image_set_loader']['train'])
val_loader = ImageSetLoader(**init_args['image_set_loader']['val'])

fcn_vgg16 = FCN(input_shape=(image_size, image_size, 3), classes=4, weight_decay=3e-3,
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
