# WRITTEN BY ALAN LI
# NASA AMES LABORATORY FOR ADVANCED SENSING (LAS)
# Last edited: Nov 8, 2017

import sys
import random
import numpy as np
import cv2
import loadcoraldata_utils as coralutils
import glob, os
from matplotlib import pyplot as plt
import matplotlib.patches as mpatches
from collections import Counter
import pandas as pd
import logging

import keras
from keras.models import Model, Sequential, load_model
from keras.layers import Input, Dense, Dropout, Flatten, Activation
from keras.layers import Conv2D, Convolution2D, MaxPooling2D
from keras.layers.convolutional import ZeroPadding2D
from keras.layers.normalization import BatchNormalization
from keras.utils import np_utils
from keras import optimizers
from keras.callbacks import Callback

def classifyback(predictions):
	return np.argmax(predictions,1)

def rescale(dataset, depth):
	dataset_norm = (dataset.astype(np.float32) - depth/2)/(depth/2)
	return dataset_norm

def load_data(Imagepath, Truthpath, truth_key = None):
	img1 = cv2.imread(Imagepath,cv2.IMREAD_UNCHANGED)
	img1_truth = cv2.imread(Truthpath,cv2.IMREAD_UNCHANGED)

	if truth_key is not None:
		item_counter = 0
		for item in truth_key:
			img1_truth[img1_truth == item ] = item_counter  # Sand
			item_counter+=1
	return img1, img1_truth.astype(np.uint8)

def load_whole_data(img1, img1_truth, image_size, depth=255, offset=0, lines=None, toremove = False):
	crop_len = int(np.floor(image_size/2))

	if lines is None:
		lines = img1.shape[0]-2*crop_len

	if offset + lines + 2*crop_len > img1.shape[0]:
		print("Too many lines specified, reverting to maximum possible")
		lines = im1.shape[0] - offset - 2*crop_len

	whole_datasets = []
	whole_labels = []
	for i in range(offset+crop_len,lines+offset+crop_len):
		for j in range(crop_len, img1.shape[1]-crop_len):
			whole_datasets.append(img1[i-crop_len:i+crop_len+1, j-crop_len:j+crop_len+1,:])
			whole_labels.append(img1_truth[i,j])

	whole_datasets = np.asarray(whole_datasets) 
	whole_labels = np.asarray(whole_labels).reshape((len(whole_labels),1))

	if toremove is not None:
		whole_datasets = np.delete(whole_datasets,toremove,-1)
	whole_dataset = rescale(whole_datasets,depth)
	return whole_dataset, whole_labels

whole_predict = []
offstart = 0
num_classes = 4
image_size = 25
crop_len = int(np.floor(image_size/2))
transect1_path = '../Images/Transect 1 Hi-Res.tiff'
transect1_truth_path = '../Images/Transect 1 Truth data.tif'
Transect1, Transect1_truth = load_data(transect1_path, transect1_truth_path, truth_key=[16,160,198,38])
num_lines = Transect1_truth.shape[0] - 2*crop_len
#num_lines = 250
model = load_model('./models/AlexNetLike.h5')

weightfiles = glob.glob('./models/*.hdf5')

for offset in range(offstart,offstart+num_lines):
	temp_dataset, temp_labelset = load_whole_data(Transect1, Transect1_truth, image_size=25, offset = offset, lines=1, toremove=3)
	temp_predict = model.predict_on_batch(temp_dataset)
	whole_predict.append(classifyback(temp_predict))
	print(str(offset+1) + '/ ' + str(num_lines) +' completed', end='\r')

whole_predict = np.asarray(whole_predict).astype(np.uint8)
whole_predict_map = np.copy(whole_predict)
whole_predict_map[whole_predict_map == 0] = -1
whole_predict_map = ((whole_predict_map+1)*255/4).astype(np.uint8)
whole_predict_map = cv2.applyColorMap(whole_predict_map, cv2.COLORMAP_JET)
cv2.imwrite('AlexNetLike_final.png',whole_predict_map)

truth_predict = Transect1_truth[crop_len+offstart:crop_len+offstart+num_lines, crop_len:Transect1_truth.shape[1]-crop_len]
accuracy = 100*(whole_predict == truth_predict).sum()/(whole_predict.shape[0]*whole_predict.shape[1])
print('Final Accuracy %.1f%%' % (accuracy))