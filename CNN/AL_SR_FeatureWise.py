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
from NeMO_models import AlexNetLike, SharpMask_FCN, TestModel, SRModel_FeatureWise
from NeMO_generator import NeMOImageGenerator, ImageSetLoader
from NeMO_backend import get_model_memory_usage
from keras.callbacks import (
    ReduceLROnPlateau,
    CSVLogger,
    EarlyStopping,
    ModelCheckpoint,
    TerminateOnNaN)
from NeMO_callbacks import CheckNumericsOps, WeightsSaver
from NeMO_losses import charbonnierLoss
import NeMO_layers
from keras.models import load_model

Jarrett_4channel_model = load_model('./tmp/RefineMask_Jarrett256_RGB_NIR2.h5', custom_objects={'BilinearUpSampling2D':NeMO_layers.BilinearUpSampling2D, 'charbonnierLoss': charbonnierLoss})

image_size = 128
batch_size = 12
mag = 2
model_name = 'SR_FeatureWise_Oct7'

jsonpath = './utils/CoralClasses.json'
with open(jsonpath) as json_file:
    json_data = json.load(json_file)

labelkey = json_data["Fiji_ClassDict"]
num_classes = len(labelkey)

with open("init_args - SRx2_Fiji.yml", 'r') as stream:
    try:
        init_args = yaml.load(stream)
    except yaml.YAMLError as exc:
        print(exc)

train_loader = ImageSetLoader(**init_args['image_set_loader']['train'])
val_loader = ImageSetLoader(**init_args['image_set_loader']['val'])

num_channels = 4 # WV2 to Sentinel hard-code

y = train_loader.target_size[1]
x = train_loader.target_size[0]
pixel_mean =100*np.ones(num_channels)
pixel_std = 100*np.ones(num_channels)
# channel_shift_range = [0.01]*num_channels
# rescale = np.asarray([[0.95,1.05]]*num_channels)

checkpointer = ModelCheckpoint(filepath="./tmp/" + model_name + ".h5", verbose=1, monitor='val_loss', mode='min', save_best_only=True)
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

# log history during model fit
csv_logger = CSVLogger('output/log.csv', append=True, separator=';')

datagen = NeMOImageGenerator(image_shape=[y, x, num_channels],
                                    image_resample=True,
                                    pixelwise_center=True,
                                    pixel_mean=pixel_mean,
                                    pixelwise_std_normalization=True,
                                    random_rotation=False,
                                    pixel_std=pixel_std,
                                    image_or_label="image")
train_generator = datagen.flow_from_NeMOdirectory(directory=[train_loader.image_dir, train_loader.label_dir],
    FCN_directory=None,
    source_size=[(x,y),(x*mag,y*mag)],
    target_size=[128],
    color_mode="4channel_delete",
    passedclasses = labelkey,
    class_mode = 'zeros',
    batch_size = batch_size,
    shuffle=True,
    image_or_label="unaltered")
    # save_to_dir='./tmpbatchsave/',
    # save_format='png',
    # image_or_label="image")

validation_generator = datagen.flow_from_NeMOdirectory(directory=[val_loader.image_dir, val_loader.label_dir],
    FCN_directory=None,
    source_size=[(x,y),(x*mag,y*mag)],
    target_size=[128],
    color_mode="4channel_delete",
    passedclasses = labelkey,
    class_mode = 'zeros',
    batch_size = batch_size,
    shuffle=True,
    image_or_label="unaltered")
    # save_to_dir='./tmpbatchsave/',
    # save_format='png',
    # image_or_label="image")


conv_layers = 3
full_layers = 0

# RCU = ("cbacb","")
# # Upsampling with NN upsampling followed by 2d conv
# conv_params = {"filters":[64,64,[64,64,4]],
#     "conv_size": [(9,9),(3,3),[(3,3),(3,3),(9,9)]],
#     "conv_strides": [(1,1), (1,1), (1,1)],
#     "padding": ['same','same', 'same'],
#     "dilation_rate": [(1,1), (1,1), (1,1)],
#     "filters_up": [None,None,64],
#     "upconv_size": [None,None,64],
#     "upconv_strides": [None, None, (2,2)],
#     "upconv_type": ["","","nn"],
#     "layercombo": ["cba", [RCU,RCU,RCU,RCU], "ucbaucbac"],
#     "layercombine": ["sum","sum","sum"]}

RCU = ("cbacb","")
# Upsampling with NN upsampling followed by 2d conv
conv_params = {"filters":[64,64,[64,64,4]],
    "conv_size": [(9,9),(3,3),[(3,3),(3,3),(9,9)]],
    "conv_strides": [(1,1), (1,1), (1,1)],
    "padding": ['same','same', 'same'],
    "dilation_rate": [(1,1), (1,1), (1,1)],
    "filters_up": [None,None,64],
    "upconv_size": [None,None,64],
    "upconv_strides": [None, None, (2,2)],
    "upconv_type": ["","","nn"],
    "layercombo": ["cba", [RCU,RCU,RCU,RCU], "ucbacbac"],
    "layercombine": ["sum","sum","sum"]}


SR_simple = AlexNetLike(input_shape=(y, x, num_channels), classes=num_classes, weight_decay=3e-3, trainable_encoder=True, weights=None, conv_layers=conv_layers, full_layers=0, conv_params=conv_params)

optimizer = keras.optimizers.Adam(1e-4)

Featurewise_SR = SRModel_FeatureWise(hr_input_shape=(x*mag,y*mag,num_channels), lr_input_shape=(y,x,num_channels), SRModel=SR_simple, FeatureModel=Jarrett_4channel_model, FeatureLayerName="add_3")
keras.utils.layer_utils.print_summary(Featurewise_SR, line_length=150, positions=[.35, .55, .65, 1.])

Featurewise_SR.compile(loss='mse', optimizer=optimizer)

print("Memory required (GB): ", get_model_memory_usage(batch_size, Featurewise_SR))

Featurewise_SR.fit_generator(train_generator,
    steps_per_epoch=400,
    epochs=100,
    validation_data=validation_generator,
    validation_steps=20,
    verbose=1,
    callbacks=[lr_reducer, early_stopper, nan_terminator, checkpointer])