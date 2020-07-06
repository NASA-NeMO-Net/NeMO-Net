import os
import yaml
import datetime
import numpy as np
import json
import keras
import keras.backend as K
from keras.losses import mean_absolute_error, mean_squared_error
from keras.models import Model, Sequential, load_model
import tensorflow as tf
import sys
sys.path.append("./utils/") # Adds higher directory to python modules path.
import NeMO_layers

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

global _SESSION
config = tf.ConfigProto(allow_soft_placement=True)
config.gpu_options.allow_growth = True
_SESSION = tf.Session(config=config)
K.set_session(_SESSION)

import sys
sys.path.append("./utils/") # Adds higher directory to python modules path.
import loadcoraldata_utils as coralutils
from NeMO_models import AlexNetLike, SharpMask_FCN, TestModel, Discriminator_AlexNetLike, SRGAN
from NeMO_generator import NeMOImageGenerator, ImageSetLoader
from NeMO_backend import get_model_memory_usage
from keras.callbacks import (
    ReduceLROnPlateau,
    CSVLogger,
    EarlyStopping,
    ModelCheckpoint,
    TerminateOnNaN)
from NeMO_callbacks import CheckNumericsOps, WeightsSaver, Learning_rate_adjuster
from NeMO_losses import charbonnierLoss, vgg19_content_loss, model_content_loss
import NeMO_layers

NeMOmodel = load_model('./tmp/RefineMask_Jarrett256_RGB_NIR_spectralshift.h5', custom_objects={'BilinearUpSampling2D':NeMO_layers.BilinearUpSampling2D, 'charbonnierLoss': charbonnierLoss})
image_size = 128
batch_size = 16
mag = 2
model_name = 'SRx2_4channel_GAN'


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

# log history during model fit
csv_logger = CSVLogger('output/log.csv', append=True, separator=';')

datagen = NeMOImageGenerator(image_shape=[y, x, num_channels],
                                    image_resample=True,
                                    pixelwise_center=True,
                                    pixel_mean=pixel_mean,
                                    pixelwise_std_normalization=True,
                                    random_rotation=True,
                                    pixel_std=pixel_std,
                                    image_or_label="image")
train_generator = datagen.flow_from_NeMOdirectory(train_loader.image_dir,
    FCN_directory=train_loader.label_dir,
    source_size=(x,y),
    target_size=(x*mag,y*mag),
    color_mode="4channel_delete",
    passedclasses = labelkey,
    class_mode = 'categorical',
    batch_size = batch_size,
    shuffle=True,
    image_or_label="image")
    # save_to_dir='./tmpbatchsave/',
    # save_format='png',
    # image_or_label="image")

validation_generator = datagen.flow_from_NeMOdirectory(val_loader.image_dir,
    FCN_directory=val_loader.label_dir,
    source_size=(x,y),
    target_size=(x*mag,y*mag),
    color_mode="4channel_delete",
    passedclasses = labelkey,
    class_mode = 'categorical',
    batch_size = batch_size,
    shuffle=True,
    image_or_label="image")
    # save_to_dir='./tmpbatchsave/',
    # save_format='png',
    # image_or_label="image")


conv_layers = 4
full_layers = 0

# Upsampling done by pixel_shuffle, which only does x2 internally
conv_params = {"filters":[64,128,256,512],
    "conv_size": [(3,3), (3,3), (3,3), (3,3)],
    "conv_strides": [[(1,1),(4,4)], [(1,1),(2,2)], [(1,1),(2,2)], [(1,1),(2,2)]],
    "padding": ['same','same','same', 'same'],
    "dilation_rate": [(1,1), (1,1), (1,1), (1,1)],
    "scaling": [1, 1, 1, 1],
    "filters_up": [None, None, None, None],
    "upconv_size": [None, None, None, None],
    "upconv_strides": [None, None, None, None],
    "upconv_type": ["", "", "", ""],
    "layercombo": ["cacba","cbacba","cbacba","cbacba"],
    "layercombine": ["", "", "", ""],
    "full_filters": [1024, 1024],
    "dropout": [0,0]}


# Load pre-trained generator first
SR_EDSR = load_model('./tmp/SRx2_Fiji_4channel_EDSR.h5', custom_objects={'PixelShuffler':NeMO_layers.PixelShuffler, 'BilinearUpSampling2D':NeMO_layers.BilinearUpSampling2D, 'charbonnierLoss': charbonnierLoss})
# Construct discriminator
Discriminator = Discriminator_AlexNetLike(input_shape=(y*mag, x*mag, num_channels), classes=num_classes, weight_decay=3e-3, trainable_encoder=True, weights=None, conv_layers=conv_layers, full_layers=0, conv_params=conv_params)
discriminator_optimizer = keras.optimizers.Adam(lr=1e-4)
Discriminator.compile(loss='binary_crossentropy', optimizer=discriminator_optimizer, metrics=[])

discriminator_lr_scheduler = Learning_rate_adjuster(step_size=100, decay=0.1, verbose=0)
discriminator_lr_scheduler.set_model(Discriminator)

GAN = SRGAN(input_shape=(y,x,num_channels), generator=SR_EDSR, discriminator=Discriminator)
gan_optimizer = keras.optimizers.Adam(lr=1e-4)
GAN.compile(loss=[model_content_loss(NeMOmodel, "add_3", [1.0, 0.0]), 'binary_crossentropy'], loss_weights=[1, 0.25], optimizer=gan_optimizer,metrics=[])
GAN_lr_scheduler = Learning_rate_adjuster(step_size=100, decay=0.1, verbose=0)
GAN_lr_scheduler.set_model(GAN)


optimizer = keras.optimizers.Adam(1e-4)

keras.utils.layer_utils.print_summary(Discriminator, line_length=150, positions=[.35, .55, .65, 1.])
keras.utils.layer_utils.print_summary(GAN, line_length=150, positions=[.35, .55, .65, 1.])

print("Memory required (GB): ", get_model_memory_usage(batch_size, Discriminator))

label_noise = 0.05
for epoch in range(30):
    GAN_lr_scheduler.on_epoch_begin(epoch)
    discriminator_lr_scheduler.on_epoch_begin(epoch)
    
    for iters in range(200):
        
        image_lr, image_hr = train_generator.next()
        image_sr = SR_EDSR.predict(image_lr)
        
        hr_labels = np.ones(batch_size) + label_noise * np.random.random(batch_size)
        sr_labels = np.zeros(batch_size) + label_noise * np.random.random(batch_size)
        
        hr_loss = Discriminator.train_on_batch(image_hr, hr_labels)
        sr_loss = Discriminator.train_on_batch(image_sr, sr_labels)
        print("Discriminator loss:", hr_loss, sr_loss)
        
        image_lr, image_hr = train_generator.next()
        labels = np.ones(batch_size)
        perceptual_loss = GAN.train_on_batch(image_lr, [image_hr, labels])
        
        print("Perceptual loss: ", perceptual_loss) # generator (total loss), MSE, Binary x-entropy
#         print(GAN.metrics_names)

    GAN_lr_scheduler.on_epoch_end(epoch)
    discriminator_lr_scheduler.on_epoch_end(epoch)
        
SR_EDSR.save("./tmp/" + model_name + ".h5")
