import os
import yaml
import datetime
import numpy as np
import json
import keras
import keras.backend as K
import tensorflow as tf

os.environ["CUDA_VISIBLE_DEVICES"] = "2"

global _SESSION
config = tf.ConfigProto(allow_soft_placement=True)
config.gpu_options.allow_growth = True
_SESSION = tf.Session(config=config)
K.set_session(_SESSION)

import sys
sys.path.append("./utils/") # Adds higher directory to python modules path.
import loadcoraldata_utils as coralutils
from NeMO_models import AlexNetLike, SharpMask_FCN, DANN_Model
from NeMO_generator import NeMOImageGenerator, ImageSetLoader, DANN_NeMODirectoryIterator
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

image_size = 256
batch_size = 8
model_name = 'VGG16_DANN256'

jsonpath = './utils/CoralClasses.json'
with open(jsonpath) as json_file:
    json_data = json.load(json_file)

labelkey = json_data["VedConsolidated_ClassDict"]
num_classes = len(labelkey)

# with open("init_args - Jarrett.yml", 'r') as stream:
#     try:
#         init_args = yaml.load(stream)
#     except yaml.YAMLError as exc:
#         print(exc)

# train_loader = ImageSetLoader(**init_args['image_set_loader']['train'])
# val_loader = ImageSetLoader(**init_args['image_set_loader']['val'])
# target_dir = '../Images/CiciaDANN_Target_256/'

# # if train_loader.color_mode == 'rgb':
# #     num_channels = 3
# # elif train_loader.color_mode == '8channel':
# #     num_channels = 8
# num_channels = 4 # hard-coded for 4 channel

# y = train_loader.target_size[1]
# x = train_loader.target_size[0]
# pixel_mean =0*np.ones(num_channels)
# pixel_std = 1*np.ones(num_channels)
# channel_shift_range = [0.01]*num_channels
# rescale = np.asarray([[0.95,1.05]]*num_channels)

# checkpointer = ModelCheckpoint(filepath="./tmp/" + model_name + ".h5", verbose=1, monitor='model_3_loss', mode='min', save_best_only=True)
# lr_reducer = ReduceLROnPlateau(monitor='model_3_loss',
#                                factor=np.sqrt(0.1),
#                                cooldown=0,
#                                patience=10, min_lr=1e-12)
# early_stopper = EarlyStopping(monitor='model_3_loss',
#                               min_delta=0.001,
#                               patience=30)
# nan_terminator = TerminateOnNaN()
# SaveWeights = WeightsSaver(filepath='./weights/', model_name=model_name, N=10)
#csv_logger = CSVLogger('output/tmp_fcn_vgg16.csv')
    #'output/{}_fcn_vgg16.csv'.format(datetime.datetime.now().isoformat()))

#check_num = CheckNumericsOps(validation_data=[np.random.random((1, 224, 224, 3)), 1],
#                             histogram_freq=100)

# log history during model fit
# csv_logger = CSVLogger('output/log.csv', append=True, separator=';')

# datagen = NeMOImageGenerator(image_shape=[y, x, num_channels],
#                                     image_resample=True,
#                                     pixelwise_center=True,
#                                     pixel_mean=pixel_mean,
#                                     pixelwise_std_normalization=True,
#                                     augmentation = 0,
#                                     channel_shift_range = 0,
#                                     random_rotation=True,
#                                     pixel_std=pixel_std)
# train_generator = datagen.flow_from_NeMOdirectory(train_loader.image_dir,
#     label_directory = train_loader.label_dir,
#     target_directory = target_dir,
#     source_size=(x,y),
#     color_mode='4channel_delete',
#     passedclasses = labelkey,
#     class_mode = 'categorical',
#     batch_size = batch_size,
# #     save_to_dir = './Generator_Outputs/',
#     shuffle=True)

# validation_generator = datagen.flow_from_NeMOdirectory(val_loader.image_dir,
#     label_directory = val_loader.label_dir,
#     target_directory = target_dir,
#     source_size=(x,y),
#     color_mode='4channel_delete',
#     passedclasses = labelkey,
#     class_mode = 'categorical',
#     batch_size = batch_size,
#     shuffle=True)

# train_generator = datagen.flow_from_NeMOdirectory(train_loader.image_dir,
#     FCN_directory = train_loader.label_dir,
#     source_size=(x,y),
#     color_mode='4channel_delete',
#     passedclasses = labelkey,
#     class_mode = 'categorical',
#     batch_size = batch_size,
# #     save_to_dir = './Generator_Outputs/',
#     shuffle=True)

# validation_generator = datagen.flow_from_NeMOdirectory(val_loader.image_dir,
#     FCN_directory = val_loader.label_dir,
#     source_size=(x,y),
#     color_mode='4channel_delete',
#     passedclasses = labelkey,
#     class_mode = 'categorical',
#     batch_size = batch_size,
#     shuffle=True)

## AlexNet for domain ----------------- 

# conv_layers = 1
# full_layers = 3

# conv_params = {"filters": [32],
#     "conv_size": [(3,3)],
#     "conv_strides": [(1,1)],
#     "padding": ["same"],
#     "dilation_rate": [(1,1)],
#     "pool_size": [None],
#     "pool_strides": [None],
#     "pad_size": [None],
#     "filters_up": [None],
#     "upconv_size": [None],
#     "upconv_strides": [None],
#     "layercombo": ["c"], 
#     "layercombine": [None],           
#     "full_filters": [64,128,2], 
#     "dropout": [0.5,0.5,0]}

# Domain_Predictor = AlexNetLike(input_shape=(128,128,64), classes=2, weight_decay=3e-3, trainable_encoder=True, weights=None, conv_layers=conv_layers, full_layers=full_layers, conv_params=conv_params)
# keras.utils.layer_utils.print_summary(Domain_Predictor,line_length=150, positions=[.35, .55, .65, 1.])


## VGG16 ----------------

conv_layers = 5
full_layers = 0

conv_params = {"filters": [64, 128, 256, 512, [512,512,512,1024,1024]],
    "conv_size": [[(1,1),(3,3)], (3,3), (3,3), (3,3), [(3,3),(3,3),(3,3),(4,4),(1,1)]],
    "conv_strides": [(1,1)]*conv_layers,
    "padding": ['same', 'same', 'same', 'same', 'same'],
    "dilation_rate": [(1,1), (1,1), (1,1), (1,1), [(1,1),(1,1),(1,1),(2,2),(1,1)]],
    "pool_size": [(2,2), (2,2), (2,2), (2,2), (2,2)],
    "pool_strides": [(2,2), (2,2), (2,2), (2,2), (2,2)],
    "pad_size": [(0,0), (0,0), (0,0), (0,0), (0,0)],
    "filters_up": [None]*conv_layers,
    "upconv_size": [None]*conv_layers,
    "upconv_strides": [None]*conv_layers,
    "layercombo": ["cbacbap","cbacbap","cbacbacbap","cbacbacbap","cbacbacbapcdcd"], 
    "layercombine": [""]*conv_layers,           
    "full_filters": [1024,1024], 
    "dropout": [0.5,0.5]}

bridge_params = {"filters": [9, 9, 9],
    "conv_size": [(1,1), (1,1), (1,1)],
    "filters_up": [None]*3,
    "upconv_size": [None]*3,
    "upconv_strides": [None]*3,
    "layercombo": ["c","c","c"],
    "layercombine": [None]*3}

prev_params = {"filters": [None]*3,
    "conv_size": [None]*3,
    "filters_up": [None]*3,
    "upconv_size": [None]*3,
    "upconv_strides": [None]*3,
    "upconv_type": [None]*3,
    "layercombo": ["","",""],
    "layercombine": ["sum","sum","sum"]} 

next_params = {"filters": [9,9,9],
    "conv_size": [None]*3,
    "pool_size": [None]*3,
    "filters_up": [9,9,9],
    "upconv_size": [(4,4),(4,4),(16,16)],
    "upconv_strides": [(2,2),(2,2),(8,8)],
    "upconv_type": ["2dtranspose","2dtranspose","2dtranspose"],
    "layercombo": ["u","u","u"],
    "layercombine": ["sum","sum","sum"]} 

decoder_index = [0,1,2]
scales= [1,1,1]

y = 256
x = 256
num_channels = 4
VGG16_DANN = SharpMask_FCN(input_shape=(y,x,num_channels), classes=num_classes, decoder_index = decoder_index, weight_decay=1e-3, trainable_encoder=True, weights=None,
    conv_layers=conv_layers, full_layers=full_layers, conv_params=conv_params, scales=scales, 
    bridge_params=bridge_params, prev_params=prev_params, next_params=next_params)
keras.utils.layer_utils.print_summary(VGG16_DANN, line_length=150, positions=[.35, .55, .65, 1.])

## Combine Models

# DANN = DANN_Model(source_input_shape=(y,x,num_channels), source_model=VGG16_DANN, domain_model=Domain_Predictor, FeatureLayerName="vgg_convblock1_pool1")
# keras.utils.layer_utils.print_summary(DANN, line_length=150, positions=[.35, .55, .65, 1.])

optimizer = keras.optimizers.Adam(1e-4)

# model 1: domain, model 3: classifier
VGG16_DANN.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
# DANN.compile(optimizer=optimizer,loss={'model_3': 'categorical_crossentropy', 'model_1': 'categorical_crossentropy'}, loss_weights={'model_3': 1.0, 'model_1': 0.1}, metrics=['accuracy'])
                                      
print("Memory required (GB): ", get_model_memory_usage(batch_size, VGG16_DANN))

# history = VGG16_DANN.fit_generator(train_generator,
#     steps_per_epoch=100,
#     epochs=10,
#     validation_data=validation_generator,
#     validation_steps=20,
#     verbose=1)
#     callbacks=[lr_reducer, early_stopper, nan_terminator, checkpointer])



