import os
import yaml
import multiprocessing.pool
import datetime
import numpy as np
import keras
import keras.backend as K
import tensorflow as tf
from keras.preprocessing.image import load_img, _list_valid_filenames_in_directory, _count_valid_files_in_directory, img_to_array
from functools import partial

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
from NeMO_losses import unsupervised_distance_loss
from NeMO_generator import NeMOImageGenerator, ImageSetLoader
from keras.callbacks import (
    ReduceLROnPlateau,
    CSVLogger,
    EarlyStopping,
    ModelCheckpoint,
    TerminateOnNaN)
from NeMO_callbacks import CheckNumericsOps, WeightsSaver

image_size = 25
batch_size = 8
model_name = 'NeMO_spectral_calibration'

imgpath = '../Images/Transect 1 Hi-Res Skew.tiff'
truthpath = '../Images/Transect 1 Truth data.tif'
Transect1 = coralutils.CoralData(imgpath, Truthpath=truthpath, truth_key=[16,160,198,38])

# labelkey = Transect1.class_labels
# labelkey = Transect1.consol_labels
labelkey = ['Sand', 'Branching', 'Mounding', 'Rock']
num_classes = Transect1.num_classes

with open("init_args - Calibration Skew.yml", 'r') as stream:
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
pixel_mean =255/2*np.ones(num_channels)
pixel_std = 255/2*np.ones(num_channels)
channel_shift_range = [0.0,0.0,0.0]
rescale = np.asarray([[1.0,1.0],[0.9,1.5],[1.0,1.0]])
save_to_dir = "./tmpbatchsave"

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
                                    pixel_std=pixel_std,
                                    channel_shift_range = channel_shift_range,
                                    NeMO_rescale = rescale)
train_generator = datagen.flow_from_NeMOdirectory(train_loader.image_dir,
    FCN_directory=train_loader.label_dir,
    target_size=(x,y),
    color_mode=train_loader.color_mode,
    classes = labelkey,
    class_mode = 'fixed_RGB',
    batch_size = batch_size,
    shuffle=True,
    save_to_dir = save_to_dir)

validation_generator = datagen.flow_from_NeMOdirectory(val_loader.image_dir,
    FCN_directory=val_loader.label_dir,
    target_size=(x,y),
    color_mode=val_loader.color_mode,
    classes = labelkey,
    class_mode = 'fixed_RGB',
    batch_size = batch_size,
    shuffle=True)

conv_layers = 1
full_layers = 1
conv_params = {"filters": [[20,48,64,80,256]],
    "conv_size": [[(4,4),(4,4),(3,3),(3,3),(3,3)]],
    "padding": [['valid','valid','valid','valid','valid']],
    "dilation_rate": [(1,1)],
    "pool_size": [(2,2)],
    "pool_strides": [(2,2)],
    "pad_size": [(0,0)],
    "layercombo": ["capcacacaca"],
    "full_filters": [3],
    "dropout": [0]}

AlexNetLike = AlexNetLike(input_shape=(y, x, num_channels), classes=num_classes, weight_decay=0.,
                weights=None, trainable_encoder=True, conv_layers=conv_layers, full_layers=full_layers, conv_params=conv_params)
optimizer = keras.optimizers.Adam(1e-4)

AlexNetLike.summary()

pool = multiprocessing.pool.ThreadPool()
white_list_formats = {'png', 'jpg', 'jpeg', 'bmp', 'ppm', 'tif'}
function_partial = partial(_count_valid_files_in_directory, white_list_formats=white_list_formats, follow_links=False)
samples = sum(pool.map(function_partial, (os.path.join(train_loader.image_dir, subdir) for subdir in labelkey)))
results = []
filenames = []
classes = np.zeros((samples,), dtype='int32')
i = 0
class_indices = dict(zip(labelkey, range(len(labelkey))))
for dirpath in (os.path.join(train_loader.image_dir, subdir) for subdir in labelkey):
    results.append(pool.apply_async(_list_valid_filenames_in_directory, (dirpath, white_list_formats, class_indices, False)))
for res in results:
    tempclasses, tempfilenames = res.get()
    classes[i:i + len(tempclasses)] = tempclasses
    filenames += tempfilenames
    i += len(tempclasses)

pool.close()
pool.join()

numfiles_per_class = int(len(filenames)/num_classes)
midpoint = int((image_size-1)/2)
class_mean = []
for i in range(num_classes):
    x = np.asarray([0,0,0], dtype=np.float64)
    for j in range(numfiles_per_class):
        img = load_img(os.path.join(train_loader.image_dir, filenames[i*numfiles_per_class+j]), grayscale=False, target_size=train_loader.target_size)
        x += img_to_array(img, data_format="channels_last")[midpoint,midpoint]
    class_mean.append(x/numfiles_per_class)

class_mean  = np.asarray(class_mean)
class_mean_transform = (class_mean-np.asarray([255/2,255/2,255/2]))*np.asarray([1/(255/2),1/(255/2),1/(255/2)])
print("class mean: ", class_mean_transform)

AlexNetLike.compile(optimizer=optimizer, loss=unsupervised_distance_loss(class_mean_transform,gamma=1), metrics=['mae'])

# print(PerosBanhos.consolclass_weights)
AlexNetLike.fit_generator(train_generator,
    steps_per_epoch=1,
    epochs=1,
    validation_data=validation_generator,
    validation_steps=10,
    verbose=1,
    callbacks=[lr_reducer, early_stopper, nan_terminator, checkpointer])