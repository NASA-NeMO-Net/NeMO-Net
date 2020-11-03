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
from NeMO_models import AlexNetLike, SharpMask_FCN, StyleTransfer
from NeMO_generator import NeMOImageGenerator, ImageSetLoader
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
model_name = 'Test_StyleTransfer'

jsonpath = './utils/CoralClasses.json'
with open(jsonpath) as json_file:
    json_data = json.load(json_file)

labelkey = json_data["VedConsolidated_ClassDict"]
num_classes = len(labelkey)

with open("init_args - Style.yml", 'r') as stream:
    try:
        init_args = yaml.load(stream)
    except yaml.YAMLError as exc:
        print(exc)

train_loader = ImageSetLoader(**init_args['image_set_loader']['train'])
val_loader = ImageSetLoader(**init_args['image_set_loader']['val'])

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

#check_num = CheckNumericsOps(validation_data=[np.random.random((1, 224, 224, 3)), 1],
#                             histogram_freq=100)

# log history during model fit
csv_logger = CSVLogger('output/log.csv', append=True, separator=';')

datagen = NeMOImageGenerator(image_shape=[y, x, num_channels],
                                    image_resample=True,
                                    pixelwise_center=True,
                                    pixel_mean=pixel_mean,
                                    pixelwise_std_normalization=True,
                                    augmentation = 1,
                                    channel_shift_range = 0,
                                    random_rotation=True,
                                    pixel_std=pixel_std,
                                    image_or_label="image")
train_generator = datagen.flow_from_NeMOdirectory(directory=[train_loader.image_dir, train_loader.label_dir],
    FCN_directory=None,
    source_size=[(x,y),(x,y)],
    target_size=[1],
    color_mode='4channel_delete',
    passedclasses = labelkey,
    class_mode = 'zeros',
    batch_size = batch_size,
#    save_to_dir = './Generator_Outputs/',
    shuffle=True,
    image_or_label="unaltered")

validation_generator = datagen.flow_from_NeMOdirectory(directory=[val_loader.image_dir, val_loader.label_dir],
    FCN_directory=None,
    source_size=[(x,y),(x,y)],
    target_size=[1],
    color_mode='4channel_delete',
    passedclasses = labelkey,
    class_mode = 'zeros',
    batch_size = batch_size,
#    save_to_dir = './Generator_Outputs/',
    image_or_label="unaltered")

conv_layers = 5
full_layers = 0

# First 4 megablocks of the resnet-50 architecture

conv_params = {"filters": [[64] , [([64,64,128],128)]*3, [([128,128,256],256)]*4, [([256,256,512],512)]*6, [([512,512,1024],1024)]*3],
    "conv_size": [[(7,7)] , [([(1,1),(3,3),(1,1)], (1,1))]*3, [([(1,1),(3,3),(1,1)], (1,1))]*4, [([(1,1),(3,3),(1,1)], (1,1))]*6, [([(1,1),(3,3),(1,1)], (1,1))]*3],
    "conv_strides": [(2,2), [([(1,1),(1,1),(1,1)], (1,1))] + [(1,1)]*2 , [([(2,2),(1,1),(1,1)], (2,2))] + [(1,1)]*3 , [([(2,2),(1,1),(1,1)], (2,2))] + [(1,1)]*5, [([(2,2),(1,1),(1,1)], (2,2))] + [(1,1)]*2],
    "padding": ['same', 'same', 'same', 'same', 'same'],
    "dilation_rate": [(1,1), (1,1), (1,1), (1,1), (1,1)],
    "pool_size": [(3,3), (1,1), (1,1), (1,1), (1,1)],
    "pool_strides": [(2,2), (1,1), (1,1), (1,1), (1,1)],
    "pad_size": [(0,0), (0,0), (0,0), (0,0), (0,0)],
    "filters_up": [None]*conv_layers,
    "upconv_size": [None]*conv_layers,
    "upconv_strides": [None]*conv_layers,
    "layercombo": ["cbap", [("cbacbac","c")]+[("bacbacbac","")]*2, [("bacbacbac","c")]+[("bacbacbac","")]*3, [("bacbacbac","c")]+[("bacbacbac","")]*5, [("bacbacbac","c")]+[("bacbacbac","")]*2], 
    "layercombine": ["","sum","sum","sum","sum"],           
    "full_filters": [1024,1024], 
    "dropout": [0,0]}

RCU = ("bacbac","")
CRPx2 = ([("pbc",""),"pbc"],"")

bridge_params = {"filters": [[1024,1024,128], [512,512,64], [256,256,32], [128,128,16]],
    "conv_size": [(3,3), (3,3), (3,3), (3,3)],
    "filters_up": [None]*4,
    "upconv_size": [None]*4,
    "upconv_strides": [None]*4,
    "layercombo": [[RCU,RCU,"bc"], [RCU,RCU,"bc"], [RCU,RCU,"bc"], [RCU,RCU,"bc"]],
    "layercombine": ["sum","sum","sum","sum"]}

prev_params = {"filters": [None, [128,128,64], [64,64,32], [32,32,16]],
    "conv_size": [None, (3,3),(3,3),(3,3)],
    "filters_up": [None,None,None,None],
    "upconv_size": [None,None,None,None],
    "upconv_strides": [None,(2,2),(2,2),(2,2)],
    "upconv_type": [None,"bilinear","bilinear","bilinear"],
    "layercombo": ["", [RCU,RCU,"bcu"], [RCU,RCU,"bcu"], [RCU,RCU,"bcu"]],
    "layercombine": [None,"sum","sum","sum"]} 

next_params = {"filters": [["",128,128], ["",64,64], ["",32,32], ["",16,16,16,None,16,None]],
    "conv_size": [(3,3), (3,3), (3,3), (3,3)],
    "pool_size": [(5,5), (5,5), (5,5), (5,5)],
    "filters_up": [None,None,None,None],
    "upconv_size": [None,None,None,None],
    "upconv_strides": [None,None,None,(2,2)],
    "upconv_type": [None,None,None,"bilinear"],
    "layercombo": [["a",CRPx2,RCU], ["a",CRPx2,RCU], ["a",CRPx2,RCU], ["a",CRPx2,RCU,RCU,"u",RCU,"u"]],
    "layercombine": ["sum","sum","sum","sum"]} 

decoder_index = [0,1,2,3]
# upsample = [False,True,True,True,True]
scales= [1,1,1,1]

RefineMask = SharpMask_FCN(input_shape=(y,x,num_channels), classes=4, decoder_index = decoder_index, weight_decay=3e-3, trainable_encoder=True, weights=None,
    conv_layers=conv_layers, full_layers=full_layers, conv_params=conv_params, scales=scales, 
    bridge_params=bridge_params, prev_params=prev_params, next_params=next_params, reshape=False)
keras.utils.layer_utils.print_summary(RefineMask, line_length=150, positions=[.25, .55, .85, 1.])

RefineMask_Jarrett256_RGB_NIR2 = load_model('./tmp/RefineMask_Jarrett256_RGB_NIR2.h5', custom_objects={'BilinearUpSampling2D':NeMO_layers.BilinearUpSampling2D, 'charbonnierLoss': charbonnierLoss}) 

# TransferModel: will undergo learning to match content/style loss
# FeatureModel: cannot be learned, imported from previous learned result
TestModel = StyleTransfer(feature_input_shape=(y,x,num_channels), product_input_shape=(y,x,num_channels), TransferModel = RefineMask, FeatureModel=RefineMask_Jarrett256_RGB_NIR2, FeatureLayers=['add_1', 'add_2'], ContentLayers=['add_3'], style_weight=0.5, variation_weight=0.1)
keras.utils.layer_utils.print_summary(TestModel, line_length=150, positions=[.25, .55, .85, 1.])

optimizer = keras.optimizers.Adam(1e-4)
TestModel.compile(loss='mse', optimizer=optimizer)

# print("Memory required (GB): ", get_model_memory_usage(batch_size, TestModel))

TestModel.fit_generator(train_generator,
    steps_per_epoch=100,
    epochs=20,
    validation_data=validation_generator,
    validation_steps=20,
    verbose=1,
    callbacks=[lr_reducer, early_stopper, nan_terminator, checkpointer])

