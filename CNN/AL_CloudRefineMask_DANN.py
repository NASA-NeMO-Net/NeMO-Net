import os
import yaml
import datetime
import numpy as np
import json
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

image_size = 80
batch_size = 8
model_name = 'CloudRefineMask_DANN80'

# jsonpath = './utils/CoralClasses.json'
# with open(jsonpath) as json_file:
#     json_data = json.load(json_file)

# labelkey = json_data["VedConsolidated_ClassDict"]
labelkey = {'No clouds': 0, 'Clouds': 1}
num_classes = 2

with open("init_args - MichalCloudsDANN.yml", 'r') as stream:
    try:
        init_args = yaml.load(stream)
    except yaml.YAMLError as exc:
        print(exc)

train_loader = ImageSetLoader(**init_args['image_set_loader']['train'])
val_loader = ImageSetLoader(**init_args['image_set_loader']['val'])
target_dir = '/home/shared/NeMO-Net Data/DANN_data/DANN_Training_S2Patches_4channel_80/'

# if train_loader.color_mode == 'rgb':
#     num_channels = 3
# elif train_loader.color_mode == '8channel':
#     num_channels = 8
num_channels = 4 # hard-coded for 4 channel

y = train_loader.target_size[1]
x = train_loader.target_size[0]
pixel_mean =0*np.ones(num_channels)
pixel_std = 1*np.ones(num_channels)
# channel_shift_range = [0.01]*num_channels
# rescale = np.asarray([[0.95,1.05]]*num_channels)

checkpointer = ModelCheckpoint(filepath="./tmp/" + model_name + ".h5", verbose=1, monitor='val_vgg_fcblock3_Dense_loss', mode='min', save_best_only=True)
lr_reducer = ReduceLROnPlateau(monitor='val_vgg_fcblock3_Dense_loss',
                               factor=np.sqrt(0.1),
                               cooldown=0,
                               patience=10, min_lr=1e-12)
early_stopper = EarlyStopping(monitor='val_vgg_fcblock3_Dense_loss',
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
                                    augmentation = 0,
                                    channel_shift_range = 0,
                                    random_rotation=True,
                                    pixel_std=pixel_std)
train_generator = datagen.DANN_flow_from_NeMOdirectory(train_loader.image_dir,
    target_directory = target_dir,
    source_size=(x,y),
    color_mode='4channel',
    passedclasses = labelkey,
    class_mode = 'categorical',
    batch_size = batch_size,
#     save_to_dir = './Generator_Outputs/',
    shuffle=True)

validation_generator = datagen.DANN_flow_from_NeMOdirectory(val_loader.image_dir,
    target_directory = val_loader.image_dir,
    source_size=(x,y),
    color_mode='4channel',
    passedclasses = labelkey,
    class_mode = 'categorical',
    batch_size = batch_size,
    shuffle=True,
    validation=True)

## VGG16 ----------------

conv_layers = 2
full_layers = 3

conv_params = {"filters": [64, 80],
    "conv_size": [(3,3),(5,5)],
    "conv_strides": [(2,2)]*conv_layers,
    "padding": ['same']*conv_layers,
    "dilation_rate": [(1,1)]*conv_layers,
    "pool_size": [None]*conv_layers,
    "pool_strides": [None]*conv_layers,
    "pad_size": [(0,0)]*conv_layers,
    "filters_up": [None]*conv_layers,
    "upconv_size": [None]*conv_layers,
    "upconv_strides": [None]*conv_layers,
    "layercombo": ["cba","cba"], 
    "layercombine": [""]*conv_layers,           
    "full_filters": [256,512,2], 
    "dropout": [0.5,0.5,0]}


Feature_Predictor = AlexNetLike(input_shape=(y,x,num_channels), classes=num_classes, weight_decay=0., trainable_encoder=True, weights=None, conv_layers=conv_layers, full_layers=full_layers, conv_params=conv_params)

keras.utils.layer_utils.print_summary(Feature_Predictor, line_length=150, positions=[.35, .55, .65, 1.])

# AlexNet for domain ----------------- 

conv_layers = 0
full_layers = 3

conv_params = {"filters": [32],
    "conv_size": [(1,1)],
    "conv_strides": [(1,1)],
    "padding": ['same'],
    "dilation_rate": [(1,1)],
    "pool_size": [None],
    "pool_strides": [None],
    "pad_size": [(0,0)],
    "filters_up": [None],
    "upconv_size": [None],
    "upconv_strides": [None],
    "layercombo": [""], 
    "layercombine": [""],           
    "full_filters": [256,256,2], 
    "dropout": [0.5,0.5,0]}

Domain_Predictor = AlexNetLike(input_shape=(20,20,80), classes=2, weight_decay=0., trainable_encoder=True, weights=None, conv_layers=conv_layers, full_layers=full_layers, conv_params=conv_params)
keras.utils.layer_utils.print_summary(Domain_Predictor,line_length=150, positions=[.35, .55, .65, 1.])

## Combine Models

DANN = DANN_Model(source_input_shape=(y,x,num_channels), source_model=Feature_Predictor, domain_model=Domain_Predictor, FeatureLayerName="vgg_convblock2_Activ1")
keras.utils.layer_utils.print_summary(DANN, line_length=150, positions=[.35, .55, .65, 1.])

optimizer = keras.optimizers.Adam(1e-3)

# vgg_fcblock3_Dense_1: domain, vgg_fcblock3_Dense: classifier
DANN.compile(optimizer=optimizer,loss={'vgg_fcblock3_Dense': 'categorical_crossentropy', 'vgg_fcblock3_Dense_1': 'categorical_crossentropy'}, loss_weights={'vgg_fcblock3_Dense': 1.0, 'vgg_fcblock3_Dense_1': 0.5}, metrics=['accuracy'])
                                       

print("Memory required (GB): ", get_model_memory_usage(batch_size, DANN))

DANN.fit_generator(train_generator,
    steps_per_epoch=10000,
    epochs=20,
    validation_data=validation_generator,
    validation_steps=20,
    verbose=1,
    callbacks=[lr_reducer, early_stopper, nan_terminator, checkpointer])



