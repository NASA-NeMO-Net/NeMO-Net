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
from NeMO_models import AlexNetLike, SharpMask_FCN
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

image_size = 80
batch_size = 16
model_name = 'VGG16FCN_Kamlika80_RGB_NIR'

jsonpath = './utils/CoralClasses.json'
with open(jsonpath) as json_file:
    json_data = json.load(json_file)

# labelkey = json_data["VedConsolidated_ClassDict"]
labelkey = {'Coral':1, 'b':2, 'c':3, 'd':4, 'e':5, 'f':6, 'g':7, 'h':8, 'i':9, 'j':10}
print(labelkey)
num_classes = len(labelkey)

with open("init_args - Kamalika.yml", 'r') as stream:
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

checkpointer = ModelCheckpoint(filepath="./tmp/" + model_name + ".h5", verbose=1, monitor='val_acc', mode='max', save_best_only=True)
lr_reducer = ReduceLROnPlateau(monitor='val_loss',
                               factor=np.sqrt(0.1),
                               cooldown=0,
                               patience=10, min_lr=1e-12)
early_stopper = EarlyStopping(monitor='val_loss',
                              min_delta=0.001,
                              patience=30)
nan_terminator = TerminateOnNaN()
SaveWeights = WeightsSaver(filepath='./weights/', model_name=model_name, N=10)

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
train_generator = datagen.flow_from_NeMOdirectory(train_loader.image_dir,
    FCN_directory=train_loader.label_dir,
    source_size=(x,y),
    target_size=(x,y),
    color_mode='4channel',
    passedclasses = labelkey,
    class_mode = 'categorical',
    batch_size = batch_size,
#    save_to_dir = './Generator_Outputs/',
    shuffle=True)

validation_generator = datagen.flow_from_NeMOdirectory(val_loader.image_dir,
    FCN_directory=val_loader.label_dir,
    source_size=(x,y),
    target_size=(x,y),
    color_mode='4channel',
    passedclasses = labelkey,
    class_mode = 'categorical',
    batch_size = batch_size,
    shuffle=True)

conv_layers = 5
full_layers = 0

# First 4 megablocks of the resnet-50 architecture
conv_params = {"filters": [64,128,256,512,1024],
    "conv_size": [(3,3),(3,3),(3,3),(2,2),(2,2)],
    "conv_strides": [(1,1),(1,1),(1,1),(1,1),(1,1)],
    "padding": ['same','same','same','same','same'],
    "dilation_rate": [(1,1),(1,1),(1,1),(1,1),(1,1)],
    "pool_size": [(2,2),(2,2),(2,2),(2,2),None],
    "pool_strides": [(2,2), (2,2), (2,2), (2,2), None],
    "pad_size": [(0,0),(0,0),(0,0),(0,0),(0,0)],
    "layercombo": ["cacapb","cacapba","cacacapb","cacacapb","cacacab"],
    "layercombine": [None]*5,
    "full_filters": [1024,1024],
    "dropout": [0,0]}

bridge_params = {"filters": [[1024,512], [512,512], [256,256], [128,128], [64,64]],
    "conv_size": [(3,3), (3,3), (3,3), (3,3), (3,3)],
    "filters_up": [None]*5,
    "upconv_size": [None]*5,
    "upconv_strides": [None]*5,
    "layercombo": ["cacab","cacab","cacab","cacab","cacab"],
    "layercombine": ["sum","sum","sum","sum","sum"]}

prev_params = {"filters": [None]*5,
    "conv_size": [None]*5,
    "filters_up": [None]*5,
    "upconv_size": [None]*5,
    "upconv_strides": [None]*5,
    "upconv_type": [None]*5,
    "layercombo": ["", "", "", "", ""],
    "layercombine": [None]*5} 

next_params = {"filters": [None]*5,
    "conv_size": [None]*5,
    "pool_size": [None]*5,
    "filters_up": [None,256,128,64,32],
    "upconv_size": [None,(2,2),(2,2),(2,2),(2,2)],
    "upconv_strides": [None,(2,2),(2,2),(2,2),(2,2)],
    "upconv_type": [None,"2dtranspose", "2dtranspose","2dtranspose","2dtranspose"],
    "layercombo": ["", "u", "u", "u", "u"],
    "layercombine": [None,"sum","sum","sum","sum"]} 


# conv_params = {"filters": [[64] , [([64,64,128],128)]*3, [([128,128,256],256)]*4, [([256,256,512],512)]*6, [([512,512,1024],1024)]*3],
#     "conv_size": [[(7,7)] , [([(1,1),(3,3),(1,1)], (1,1))]*3, [([(1,1),(3,3),(1,1)], (1,1))]*4, [([(1,1),(3,3),(1,1)], (1,1))]*6, [([(1,1),(3,3),(1,1)], (1,1))]*3],
#     "conv_strides": [(2,2), [([(1,1),(1,1),(1,1)], (1,1))] + [(1,1)]*2 , [([(2,2),(1,1),(1,1)], (2,2))] + [(1,1)]*3 , [([(2,2),(1,1),(1,1)], (2,2))] + [(1,1)]*5, [([(2,2),(1,1),(1,1)], (2,2))] + [(1,1)]*2],
#     "padding": ['same', 'same', 'same', 'same', 'same'],
#     "dilation_rate": [(1,1), (1,1), (1,1), (1,1), (1,1)],
#     "pool_size": [(3,3), (1,1), (1,1), (1,1), (1,1)],
#     "pool_strides": [(2,2), (1,1), (1,1), (1,1), (1,1)],
#     "pad_size": [(0,0), (0,0), (0,0), (0,0), (0,0)],
#     "filters_up": [None]*conv_layers,
#     "upconv_size": [None]*conv_layers,
#     "upconv_strides": [None]*conv_layers,
#     "layercombo": ["cbap", [("cbacbac","c")]+[("bacbacbac","")]*2, [("bacbacbac","c")]+[("bacbacbac","")]*3, [("bacbacbac","c")]+[("bacbacbac","")]*5, [("bacbacbac","c")]+[("bacbacbac","")]*2], 
#     "layercombine": ["","sum","sum","sum","sum"],           
#     "full_filters": [1024,1024], 
#     "dropout": [0,0]}

# RCU = ("bacbac","")
# CRPx2 = ([("pbc",""),"pbc"],"")

# bridge_params = {"filters": [[1024,1024,128], [512,512,64], [256,256,32], [128,128,16]],
#     "conv_size": [(3,3), (3,3), (3,3), (3,3)],
#     "filters_up": [None]*4,
#     "upconv_size": [None]*4,
#     "upconv_strides": [None]*4,
#     "layercombo": [[RCU,RCU,"bc"], [RCU,RCU,"bc"], [RCU,RCU,"bc"], [RCU,RCU,"bc"]],
#     "layercombine": ["sum","sum","sum","sum"]}

# prev_params = {"filters": [None, [128,128,64], [64,64,32], [32,32,16]],
#     "conv_size": [None, (3,3),(3,3),(3,3)],
#     "filters_up": [None,None,None,None],
#     "upconv_size": [None,None,None,None],
#     "upconv_strides": [None,(2,2),(2,2),(2,2)],
#     "upconv_type": [None,"bilinear","bilinear","bilinear"],
#     "layercombo": ["", [RCU,RCU,"bcu"], [RCU,RCU,"bcu"], [RCU,RCU,"bcu"]],
#     "layercombine": [None,"sum","sum","sum"]} 

# next_params = {"filters": [["",128,128], ["",64,64], ["",32,32], ["",16,16,16,None,16,None]],
#     "conv_size": [(3,3), (3,3), (3,3), (3,3)],
#     "pool_size": [(5,5), (5,5), (5,5), (5,5)],
#     "filters_up": [None,None,None,None],
#     "upconv_size": [None,None,None,None],
#     "upconv_strides": [None,None,None,(2,2)],
#     "upconv_type": [None,None,None,"bilinear"],
#     "layercombo": [["a",CRPx2,RCU], ["a",CRPx2,RCU], ["a",CRPx2,RCU], ["a",CRPx2,RCU,RCU,"u",RCU,"u"]],
#     "layercombine": ["sum","sum","sum","sum"]} 

decoder_index = [0,1,2,3,4]
# upsample = [False,True,True,True,True]
scales= [1,1,1,1,1]

VGG16_FCN = SharpMask_FCN(input_shape=(y,x,num_channels), classes=num_classes, decoder_index = decoder_index, weight_decay=3e-3, trainable_encoder=True, weights=None,
    conv_layers=conv_layers, full_layers=full_layers, conv_params=conv_params, scales=scales, 
    bridge_params=bridge_params, prev_params=prev_params, next_params=next_params)

optimizer = keras.optimizers.Adam(1e-4)

# SharpMask.summary()
keras.utils.layer_utils.print_summary(VGG16_FCN, line_length=150, positions=[.35, .55, .65, 1.])
VGG16_FCN.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'], sample_weight_mode='temporal')

print("Memory required (GB): ", get_model_memory_usage(batch_size, VGG16_FCN))

# VGG16_FCN.fit_generator(train_generator,
#     steps_per_epoch=100,
#     epochs=100,
#     validation_data=validation_generator,
#     validation_steps=20,
#     verbose=1,
#     callbacks=[lr_reducer, early_stopper, nan_terminator, checkpointer])

