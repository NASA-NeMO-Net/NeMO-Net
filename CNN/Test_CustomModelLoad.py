# WRITTEN BY ALAN LI
# NASA AMES LABORATORY FOR ADVANCED SENSING (LAS)
# Last edited: Feb 22, 2018

import sys
sys.path.append("./utils/") # Adds higher directory to python modules path.
import random
import numpy as np
import cv2
import glob, os
from collections import Counter

from matplotlib import pyplot as plt
import matplotlib.patches as mpatches
from matplotlib import colors
from collections import Counter
import pandas as pd
import logging
import yaml
from PIL import Image as pil_image

import tensorflow as tf
from tensorflow.python.client import device_lib
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import loadcoraldata_utils as coralutils
import keras
import keras.backend as K
from keras.models import Model, Sequential, load_model
from keras.utils import np_utils
from keras import optimizers
from keras.preprocessing.image import img_to_array
from keras.callbacks import Callback
from NeMO_models import AlexNetLike
import NeMO_layers
import NeMO_encoders
from NeMO_generator import NeMOImageGenerator, ImageSetLoader
from NeMO_callbacks import CheckNumericsOps, WeightsSaver

num_cores = 4
num_GPU = 1
num_CPU = 1

global _SESSION
config = tf.ConfigProto(intra_op_parallelism_threads=num_cores,
        inter_op_parallelism_threads=num_cores, allow_soft_placement=True,
        device_count = {'CPU' : num_CPU, 'GPU' : num_GPU})
_SESSION = tf.Session(config=config)
K.set_session(_SESSION)

# model = load_model('./tmp/VGG16DeepLab_Raster256.h5', 
#                    custom_objects={'BilinearUpSampling2D':NeMO_layers.BilinearUpSampling2D})

conv_layers = 2
full_layers = 0
num_classes = 8

conv_params = {"filters": [(64,64,64), [([64,(64,64)],128),128]],
    "conv_size": [(2,2)],
    "conv_strides": [(1,1)],
    "padding": ['same'],
    "dilation_rate": [(1,1)],
    "pad_size": [(0,0)],
    "layercombo": [("cba","cba","cba"), [(["cba",("cba","cba")],"cba"),"c"]],
    "layercombine": ["cat",["sum","cat"]],           
    "full_filters": [1024,1024],
    "dropout": [0,0]}

TestModel = AlexNetLike(input_shape=(256, 256, 3), classes=num_classes, weight_decay=3e-3, trainable_encoder=True, 
                weights=None, conv_layers=conv_layers, full_layers=0, conv_params=conv_params)
optimizer = keras.optimizers.Adam(1e-4)

TestModel.summary()

