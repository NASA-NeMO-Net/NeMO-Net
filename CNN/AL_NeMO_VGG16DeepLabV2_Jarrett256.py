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
from NeMO_models import VGG16_DeepLabV2
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
batch_size = 5
model_name = 'VGG16DeepLab_Jarrett256'

jsonpath = './utils/CoralClasses.json'
with open(jsonpath) as json_file:
    json_data = json.load(json_file)

# imgpath = '../Images/BTPB-WV2-2012-15-8Band-mosaic-GeoTiff-Sample-AOI/BTPB-WV2-2012-15-8Band-mosaic-GeoTiff-Sample-AOI.tif'
# tfwpath = '../Images/BTPB-WV2-2012-15-8Band-mosaic-GeoTiff-Sample-AOI/BTPB-WV2-2012-15-8Band-mosaic-GeoTiff-Sample-AOI.tfw'
# truthpath = '../Images/BIOT-PerosBanhos-sample-habitat-map/BIOT-PerosBanhos-sample-habitat-map.shp'
# PerosBanhos = coralutils.CoralData(imgpath, Truthpath=truthpath, load_type="raster", tfwpath=tfwpath)
# PerosBanhos.load_PB_consolidated_classes()

labelkey = json_data["VedConsolidated_ClassDict"]
num_classes = len(labelkey)
# class_weights = {'Reef Crest - coralline algae ridge': 263.83293861316076, 'Fore-reef deep slope': 70.06211128410703, 'Fore-reef shallow slope': 151.81276890771184, 
#     'Fore-reef shallow terrace': 190.9951447921521, 'Fore-reef octocorals-dominated (Caribbean)': 0, 'Back-reef pavement': 23.805211835880574, 
#     'Back-reef coral framework': 504.06732475209594, 'Back-reef coral bommies': 767.9690461410341, 'Back-reef octocorals-dominated (Caribbean)': 0, 
#     'Lagoon Pinnacle reefs': 190.52354241167194, 'Lagoon Patch reefs': 2340.5541997121954, 'Lagoon Fringing reefs': 121.64149633228104, 'Lagoon Deep water': 114.45460804976489, 
#     'Fore-reef sand flats': 372.9389698456479, 'Back-reef sediment-dominated': 500.2607762487424, 'Lagoon sediment apron - Barren': 4.634798906159125, 
#     'Terrestrial Vegetated': 13.120074519621062, 'Terrestrial Mangroves': 1099.7360264110896, 'Intertidal Wetlands': 4164.279931145626, 'Beach (sand)': 1230.0551785450966, 
#     'Beach (rock)': 0, 'Seagrass Meadows': 263.06976897868566, 'Deep Ocean Water': 2.060391359758762, 'Other': 8.73977287635436}
# labelkey = PerosBanhos.consol_labels
# num_classes = len(PerosBanhos.PB_consolidated_classes)

with open("init_args - Jarrett.yml", 'r') as stream:
    try:
        init_args = yaml.load(stream)
    except yaml.YAMLError as exc:
        print(exc)

train_loader = ImageSetLoader(**init_args['image_set_loader']['train'])
val_loader = ImageSetLoader(**init_args['image_set_loader']['val'])

if train_loader.color_mode == 'rgb':
    num_channels = 3
elif train_loader.color_mode == '8channel':
    num_channels = 8

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
                                    pixel_std=pixel_std)
train_generator = datagen.flow_from_NeMOdirectory(train_loader.image_dir,
    FCN_directory=train_loader.label_dir,
    target_size=(x,y),
    color_mode=train_loader.color_mode,
    passedclasses = labelkey,
    class_mode = 'categorical',
    batch_size = batch_size,
    shuffle=True)

validation_generator = datagen.flow_from_NeMOdirectory(val_loader.image_dir,
    FCN_directory=val_loader.label_dir,
    target_size=(x,y),
    color_mode=val_loader.color_mode,
    passedclasses = labelkey,
    class_mode = 'categorical',
    batch_size = batch_size,
    shuffle=True)

conv_layers = 5
full_layers = 0

conv_params = {"filters": [64,128,256,512,512],
    "conv_size": [(3,3),(3,3),(3,3),(3,3),(3,3)],
    "conv_strides": [(1,1),(1,1),(1,1),(1,1),(1,1)],
    "padding": ['same','same','same','same','same'],
    "dilation_rate": [(1,1),(1,1),(1,1),(1,1),(2,2)],
    "pool_size": [(2,2),(2,2),(2,2),(3,3),(3,3)],
    "pool_strides": [(2,2),(2,2),(2,2),(1,1),(1,1)],
    "pad_size": [(0,0),(0,0),(1,1),(1,1),(1,1)],
    "layercombo": ["cacapb","cacapb","cacacapb","cacacazpb","cacacazp"],
    "full_filters": [1024,1024],
    "dropout": [0,0]}

parallelconv_params = {"filters": [[1024,1024,num_classes]],
    "conv_size": [[(3,3),(1,1),(1,1)]],
    "conv_strides":  [[(1,1),(1,1),(1,1)]],
    "padding": ['valid','valid','valid','valid'],
    "dilation_rate": [[(6,6),(1,1),(1,1)], [(12,12),(1,1),(1,1)], [(18,18),(1,1),(1,1)], [(24,24),(1,1),(1,1)]],
    "pool_size": [[(2,2),(2,2),(2,2)]], #doesn't matter
    "pool_strides": [[(2,2),(2,2),(2,2)]], #doesn't matter
    "pad_size": [[(6,6),(0,0),(0,0)], [(12,12),(0,0),(0,0)], [(18,18),(0,0),(0,0)], [(24,24),(0,0),(0,0)]],
    "layercombo": ["zcadcadc","zcadcadc","zcadcadc","zcadcadc"],
    "full_filters": [4096,2048],
    "dropout": [0.5,0.5]}

# If loading from previous model
# VGG_DeepLab = load_model('./tmp/VGG16DeepLab_Fiji256.h5', custom_objects={'BilinearUpSampling2D':NeMO_layers.BilinearUpSampling2D})
# If starting new model
VGG16_DeepLab = VGG16_DeepLabV2(input_shape=(y, x, num_channels), classes=num_classes, weight_decay=3e-3, batch_size=batch_size, 
                weights=None, trainable_encoder=True, conv_layers=5, full_layers=0, conv_params=conv_params, parallel_layers=4, parallelconv_params=parallelconv_params)
optimizer = keras.optimizers.Adam(1e-4)

VGG16_DeepLab.summary()
VGG16_DeepLab.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'], sample_weight_mode='temporal')

# print("Memory required (GB): ", get_model_memory_usage(batch_size, VGG16_DeepLab))

VGG16_DeepLab.fit_generator(train_generator,
    steps_per_epoch=10,
    epochs=100,
    validation_data=validation_generator,
    validation_steps=5,
    verbose=1,
    callbacks=[lr_reducer, early_stopper, nan_terminator, checkpointer])