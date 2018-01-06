import os
import yaml
import datetime
import numpy as np
import keras
import keras.backend as K
import tensorflow as tf
import sys
sys.path.append("./utils/") # Adds higher directory to python modules path.
from NeMO_models import FCN, ResNet34, AlexNet
from NeMO_generator import NeMOImageGenerator, ImageSetLoader
from keras.callbacks import (
    ReduceLROnPlateau,
    CSVLogger,
    EarlyStopping,
    ModelCheckpoint,
    TerminateOnNaN)
from NeMO_callbacks import CheckNumericsOps

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

global _SESSION
config = tf.ConfigProto(allow_soft_placement=True)
config.gpu_options.allow_growth = True
_SESSION = tf.Session(config=config)
K.set_session(_SESSION)

image_size = 25

with open("init_args - AlexNet.yml", 'r') as stream:
    try:
        init_args = yaml.load(stream)
    except yaml.YAMLError as exc:
        print(exc)

checkpointer = ModelCheckpoint(filepath="./tmp/AlexNet_weights.h5", verbose=1, save_best_only=True)
lr_reducer = ReduceLROnPlateau(monitor='val_loss',
                               factor=np.sqrt(0.1),
                               cooldown=0,
                               patience=10, min_lr=1e-12)
early_stopper = EarlyStopping(monitor='val_loss',
                              min_delta=0.001,
                              patience=30)
nan_terminator = TerminateOnNaN()


datagen = NeMOImageGenerator(image_shape=[image_size, image_size, 3],
                                    image_resample=True,
                                    pixelwise_center=True,
                                    pixel_mean=[127, 127, 127],
                                    pixelwise_std_normalization=True,
                                    pixel_std=[127, 127, 127])

# train_loader = ImageSetLoader(**init_args['image_set_loader']['train'])
# val_loader = ImageSetLoader(**init_args['image_set_loader']['val'])

alexnet = AlexNet(input_shape=(image_size, image_size, 3), classes=4, weight_decay=3e-3, weights=None, trainable_encoder=True)
alexnet.summary()
optimizer = keras.optimizers.Adam(1e-4)

alexnet.compile(optimizer=optimizer,
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

tempbatchsave_dir = './tmpbatchsave/'
train_generator = datagen.flow_from_NeMOdirectory('../Images/Training_Patches/',
  target_size=(image_size, image_size),
  color_mode='rgb',
  classes=['Sand','Branching','Mounding','Rock'],
  class_mode='categorical',
  batch_size=64,
  shuffle=True)

valid_generator = datagen.flow_from_directory('../Images/Valid_Patches/',
  target_size=(image_size, image_size),
  color_mode='rgb',
  classes=['Sand','Branching','Mounding','Rock'],
  class_mode='categorical',
  batch_size=64,
  shuffle=True)

alexnet.fit_generator(train_generator,
  steps_per_epoch=625,
  epochs=10,
  validation_data=valid_generator,
  validation_steps=5,
  verbose=1,
  callbacks=[lr_reducer, early_stopper, nan_terminator, checkpointer])

alexnet.save('./tmp/alexnet_model.h5')
