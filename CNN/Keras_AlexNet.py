# WRITTEN BY ALAN LI
# NASA AMES LABORATORY FOR ADVANCED SENSING (LAS)
# Last edited: Oct 2, 2017

import sys
import random
import numpy as np
import cv2
import loadcoraldata_utils as coralutils
import glob
from matplotlib import pyplot as plt
import matplotlib.patches as mpatches
from collections import Counter
import pandas as pd
import logging

import keras
from keras.models import Model, Sequential
from keras.layers import Input, Dense, Dropout, Flatten, Activation
from keras.layers import Conv2D, Convolution2D, MaxPooling2D
from keras.layers.convolutional import ZeroPadding2D
from keras.layers.normalization import BatchNormalization
from keras.utils import np_utils
from keras import optimizers
from keras.callbacks import Callback

class WeightsSaver(Callback):
    def __init__(self, filepath, N):
    	super(WeightsSaver, self).__init__()
    	self.N = N
    	self.batch = 0
    	self.epoch = 0
    	self.filepath = filepath

    def on_batch_end(self, batch, logs={}):
    	if self.batch % self.N == 0:
    		name = 'weights_epoch%02d_batch%08d.hdf5' % (self.epoch, self.batch)
    		savestr = self.filepath+name
    		self.model.save_weights(savestr, overwrite=True)
    	self.batch += 1

    def on_epoch_end(self, epoch, logs=None):
    	self.epoch += 1

transect1_path = '../Images/Transect 1 Hi-Res.tiff'
transect1_truth_path = '../Images/Transect 1 Truth data.tif'

image_size = 25

Transect1 = coralutils.CoralData(transect1_path, Truthpath=transect1_truth_path, truth_key=[16,160,198,38])
Transect1.generate_trainingset(image_size=image_size, N_train=20000, idxremove = 3, figureson = False)
Transect1.generate_validset(image_size=image_size, N_valid=2500, idxremove= 3, figureson = False)
Transect1.generate_testset(image_size=image_size, N_test=2500, idxremove = 3, figureson = False)

if Transect1.train_labels.shape[-1] != Transect1.num_classes:
    Transect1.train_labels = keras.utils.to_categorical(Transect1.train_labels, Transect1.num_classes)
    Transect1.valid_labels = keras.utils.to_categorical(Transect1.valid_labels, Transect1.num_classes)
    Transect1.test_labels = keras.utils.to_categorical(Transect1.test_labels, Transect1.num_classes)

print(Transect1.train_datasets.shape, Transect1.train_labels.shape)
print(Transect1.valid_datasets.shape, Transect1.valid_labels.shape)
print(Transect1.test_datasets.shape, Transect1.test_labels.shape)
	
DROPOUT = 0.5
WEIGHT_DECAY = 0.005
MOMENTUM = 0.9
LEARNING_RATE = 0.001
model_input = Input(shape = (image_size, image_size, 3))

# Model parameters
conv1_size, conv1_stride, pool1_size , pool1_stride = 5,1,2,2
conv2_size, conv2_stride, pool2_size , pool2_stride = 5,1,2,2
conv3_size, conv3_stride = 3,1
conv4_size, conv4_stride = 3,1
conv5_size, conv5_stride, pool5_size, pool5_stride = 3,1,2,2

filters1 = 128
filters2 = 256
# filters3 = 256
#filters4 = 384
filters5 = 512
full1 = 1024
full2 = 512

# 1st convolutional Layer
z = Conv2D(filters = filters1, kernel_size = (conv1_size,conv1_size), strides = (conv1_stride,conv1_stride), activation = "relu")(model_input)
z = MaxPooling2D(pool_size = (pool1_size,pool1_size), strides=(pool1_stride,pool1_stride))(z)
z = BatchNormalization()(z)

# 2nd convolutional Layer
z = Conv2D(filters = filters2, kernel_size = (conv2_size,conv2_size), strides = (conv2_stride,conv2_stride), activation = "relu")(z)
z = MaxPooling2D(pool_size = (pool2_size,pool2_size), strides=(pool2_stride,pool2_stride))(z)
z = BatchNormalization()(z)

# 3rd convolutional layer
# z = ZeroPadding2D(padding = (1,1))(z)
# z = Conv2D(filters = filters3, kernel_size = (conv3_size,conv3_size), strides = (conv3_stride,conv3_stride), activation = "relu")(z)

# 4th convolutional layer
# z = ZeroPadding2D(padding = (1,1))(z)
# z = Conv2D(filters = filters4, kernel_size = (conv4_size,conv4_size), strides = (conv4_stride,conv4_stride), activation = "relu")(z)

# 5th convolutional layer
#z = ZeroPadding2D(padding = (1,1))(z)
z = Conv2D(filters = filters5, kernel_size = (conv5_size,conv5_size), strides = (conv5_stride,conv5_stride), activation = "relu")(z)
# z = MaxPooling2D(pool_size = (pool5_size,pool5_size), strides=(pool5_stride,pool5_stride))(z)
z = Flatten()(z)

# 6th fully connected layer
z = Dense(full1, activation="relu")(z)
# z = Dropout(DROPOUT)(z)

# 7th fully connected layer
z = Dense(full2, activation="relu")(z)
# z = Dropout(DROPOUT)(z)

model_output = Dense(Transect1.num_classes, activation = 'softmax')(z)

model = Model(model_input, model_output)
model.summary()

batch_size = 64
epochs = 2

model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=optimizers.SGD(lr=LEARNING_RATE, momentum = MOMENTUM, decay = WEIGHT_DECAY),
              metrics=['accuracy'])

SaveWeights = WeightsSaver(filepath='./models/', N=50)
model.fit(Transect1.train_datasets, Transect1.train_labels,
          batch_size=batch_size,
          epochs=epochs,
          verbose=1,
          validation_data=(Transect1.valid_datasets, Transect1.valid_labels),
          callbacks=[SaveWeights])
score = model.evaluate(Transect1.test_datasets, Transect1.test_labels, verbose=0)

model.save('./models/AlexNetLike.h5')
print('Test loss:', score[0])
print('Test accuracy:', score[1])