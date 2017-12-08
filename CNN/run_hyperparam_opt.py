import os
import datetime
import numpy as np
import keras
import keras.backend as K
import tensorflow as tf
import sys
sys.path.append("./utils/")
import cv2
import loadcoraldata_utils as coralutils
sys.path.append("./hyperparamopt_utils/")
from train2opt import TrainOptimizer
from NeMO_models import FCN
from NeMO_generator import NeMOImageGenerator, ImageSetLoader
from keras.callbacks import (
    ReduceLROnPlateau,
    CSVLogger,
    EarlyStopping,
    ModelCheckpoint,
    TerminateOnNaN)
from NeMO_callbacks import CheckNumericsOps, WeightsSaver
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials

optModel = TrainOptimizer(labelkey=('Sand', 'Branching', 'Mounding', 'Rock'), train_image_path='../Images/Training_Patches/',
              train_label_path = '../Images/TrainingRef_Patches/', train_out_file = 'NeMO_train.txt',
              valid_image_path = '../Images/Valid_Patches/', valid_label_path = '../Images/ValidRef_Patches/',
              valid_out_file = 'NeMO_valid.txt', pixel_mean = [127.5, 127.5, 127.5], pixel_std = [127.5, 127.5, 127.5],
              num_classes = 4, model = FCN, model_name = "NeMO_FCN")

param_space=optModel.gen_param_space()
print(param_space)
print("lr=",param_space['lr'])

global _SESSION
config = tf.ConfigProto(allow_soft_placement=True)
config.gpu_options.allow_growth = True
_SESSION = tf.Session(config=config)
K.set_session(_SESSION)

test_model = optModel.model2opt_test(param_space)