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
from NeMO_models import AlexNetLike
from NeMO_generator import NeMOImageGenerator, ImageSetLoader
from NeMO_backend import get_model_memory_usage
from keras.callbacks import (
    ReduceLROnPlateau,
    CSVLogger,
    EarlyStopping,
    ModelCheckpoint,
    TerminateOnNaN)
from NeMO_callbacks import CheckNumericsOps, WeightsSaver

image_size = 256
batch_size = 8
model_name = 'VGG16DeepLab_Jarrett256_RGB_NIR_spectralshift'

jsonpath = './utils/CoralClasses.json'
with open(jsonpath) as json_file:
    json_data = json.load(json_file)

labelkey = json_data["VedConsolidated_ClassDict"]
num_classes = len(labelkey)

with open("init_args - Jarrett.yml", 'r') as stream:
    try:
        init_args = yaml.load(stream)
    except yaml.YAMLError as exc:
        print(exc)

train_loader = ImageSetLoader(**init_args['image_set_loader']['train'])
val_loader = ImageSetLoader(**init_args['image_set_loader']['val'])

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
                                    augmentation=1,
                                    channel_shift_range = 0,
                                    random_rotation=True,
                                    pixel_std=pixel_std)
train_generator = datagen.flow_from_NeMOdirectory(train_loader.image_dir,
    FCN_directory=train_loader.label_dir,
    source_size=(x,y),                                              
    target_size=(x,y),
    color_mode='4channel_delete',
    passedclasses = labelkey,
    class_mode = 'categorical',
    batch_size = batch_size,
    shuffle=True)

validation_generator = datagen.flow_from_NeMOdirectory(val_loader.image_dir,
    FCN_directory=val_loader.label_dir,
    source_size=(x,y),                                                   
    target_size=(x,y),                    
    color_mode='4channel_delete',
    passedclasses = labelkey,
    class_mode = 'categorical',
    batch_size = batch_size,
    shuffle=True)

conv_layers = 7
full_layers = 0

conv_params = {"filters": [64,128,256,512,512, ([512,512,num_classes],[512,512,num_classes],[512,512,num_classes],[512,512,num_classes]),num_classes],
    "conv_size": [(3,3),(3,3),(3,3),(3,3),(3,3), ([(3,3),(1,1),(1,1)],[(3,3),(1,1),(1,1)],[(3,3),(1,1),(1,1)],[(3,3),(1,1),(1,1)]),(1,1)],
    "conv_strides": [(1,1),(1,1),(1,1),(1,1),(1,1), ([(1,1),(1,1),(1,1)],[(1,1),(1,1),(1,1)],[(1,1),(1,1),(1,1)],[(1,1),(1,1),(1,1)]),(1,1)],
    "padding": ['same','same','same','same','same','valid','same'],
    "dilation_rate": [(1,1),(1,1),(1,1),(1,1),(2,2),([(6,6),(1,1),(1,1)],[(12,12),(1,1),(1,1)],[(18,18),(1,1),(1,1)],[(24,24),(1,1),(1,1)]),(1,1)],
    "pool_size": [(2,2),(2,2),(2,2),(3,3),(3,3),(2,2),(1,1)], #last element doesn't matter
    "pool_strides": [(2,2),(2,2),(2,2),(1,1),(1,1),(1,1),(1,1)], #last element doesn't matter
    "pad_size": [(0,0),(0,0),(0,0),(0,0),(0,0),([(6,6),(0,0),(0,0)],[(12,12),(0,0),(0,0)],[(18,18),(0,0),(0,0)],[(24,24),(0,0),(0,0)]),(0,0)],
    "filters_up": [None,None,None,None,None,None,None],
    "upconv_size": [None,None,None,None,None,None,None],
    "upconv_strides": [None,None,None,None,None,None,(8,8)],
    "upconv_type": [None,None,None,None,None,None,"bilinear"],
    "layercombo": ["cbacbap","cbacbap","cbacbacbap","cbacbacbap","cbacbacbap",("zcbacbac","zcbacbac","zcbacbac","zcbacbac"),"u"],
    "layercombine": [None,None,None,None,None,"cat",None],  
    "full_filters": [1024,1024],
    "dropout": [0,0]}

# parallelconv_params = {"filters": [[1024,1024,num_classes]],
#     "conv_size": [[(3,3),(1,1),(1,1)]],
#     "conv_strides":  [[(1,1),(1,1),(1,1)]],
#     "padding": ['valid','valid','valid','valid'],
#     "dilation_rate": [[(6,6),(1,1),(1,1)], [(12,12),(1,1),(1,1)], [(18,18),(1,1),(1,1)], [(24,24),(1,1),(1,1)]],
#     "pool_size": [[(2,2),(2,2),(2,2)]], #doesn't matter
#     "pool_strides": [[(2,2),(2,2),(2,2)]], #doesn't matter
#     "pad_size": [[(6,6),(0,0),(0,0)], [(12,12),(0,0),(0,0)], [(18,18),(0,0),(0,0)], [(24,24),(0,0),(0,0)]],
#     "layercombo": ["zcadcadc","zcadcadc","zcadcadc","zcadcadc"],
#     "full_filters": [4096,2048],
#     "dropout": [0.5,0.5]}

# If loading from previous model
# VGG_DeepLab = load_model('./tmp/VGG16DeepLab_Fiji256.h5', custom_objects={'BilinearUpSampling2D':NeMO_layers.BilinearUpSampling2D})
# If starting new model

VGG16_DeepLab = AlexNetLike(input_shape=(y,x,num_channels), classes=num_classes, weight_decay=3e-3, trainable_encoder=True, weights=None, conv_layers=conv_layers, full_layers=0, conv_params=conv_params)

# VGG16_DeepLab = VGG16_DeepLabV2(input_shape=(y, x, num_channels), classes=num_classes, weight_decay=3e-3, batch_size=batch_size, 
#                 weights=None, trainable_encoder=True, conv_layers=5, full_layers=0, conv_params=conv_params, parallel_layers=4, parallelconv_params=parallelconv_params)
optimizer = keras.optimizers.Adam(1e-4)
keras.utils.layer_utils.print_summary(VGG16_DeepLab, line_length=150, positions=[.35, .55, .65, 1.])

VGG16_DeepLab.summary()
VGG16_DeepLab.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'], sample_weight_mode='temporal')

print("Memory required (GB): ", get_model_memory_usage(batch_size, VGG16_DeepLab))

VGG16_DeepLab.fit_generator(train_generator,
    steps_per_epoch=100,
    epochs=100,
    validation_data=validation_generator,
    validation_steps=20,
    verbose=1,
    callbacks=[lr_reducer, early_stopper, nan_terminator, checkpointer])

    # RefineMask.fit_generator(train_generator,
#     steps_per_epoch=100,
#     epochs=100,
#     validation_data=validation_generator,
#     validation_steps=20,
#     verbose=1,
#     callbacks=[lr_reducer, early_stopper, nan_terminator, checkpointer])