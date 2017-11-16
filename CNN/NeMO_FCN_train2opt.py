import os
import yaml
import datetime
import numpy as np
import keras
import keras.backend as K
import tensorflow as tf
import sys
sys.path.append("./utils/") # Adds higher directory to python modules path.
import loadcoraldata_utils as coralutils
#sys.path.append("./tmp/")
from NeMO_models import FCN
from NeMO_generator import NeMOImageGenerator, ImageSetLoader
from keras.callbacks import (
    ReduceLROnPlateau,
    CSVLogger,
    EarlyStopping,
    ModelCheckpoint,
    TerminateOnNaN)
from NeMO_callbacks import CheckNumericsOps, WeightsSaver


# Class of models to be trained using optimization scheme
class TrainOptimizer:
     """An optimizer method class for training and optimizing models"""

    def __init__(self,
                 image_shape=(100, 100, 3),
                 row_patch_size = 100,
                 col_patch_size = 100,
                 image_size     = 100,
                 num_clzasses   =0,
                 weight_decay   =3e-3,
                 lr=1e-4,
                 model = None,
                 labelkey = None,
                 train_image_path=None,
                 train_label_path=None,
                 train_out_file = None,
                 trainSample    = 200,
                 valid_image_path=None,
                 valid_label_path=None,
                 valid_out_file = None,
                 validSample    = 20,
                 test_image_path= None,
                 test_label_path= None,
                 test_out_file  = None,
                 testSample     = 20,
				 ymlfile        = None):
        """Init."""
        self.image_shape     = tuple(image_shape)
        self.row_patch_size  = row_patch_size
        self.col_patch_size  = col_patch_size
        self.image_size      = image_sizelabelkey
        #self.pixelwise_center = pixelwise_center
        #self.pixel_mean = np.array(pixel_mean)
        #self.pixelwise_std_normalization = pixelwise_std_normalization
        #self.pixel_std = np.array(pixel_std)
        #super(train2opt, self).__init__()



#### generate data for train/test
# Input:
# 	from train2opt self
    def gen_data(self):
        # generate train set
        if train_image_path is not None:
             export_segmentation_map(train_image_path, train_label_path, train_out_file, image_size=image_size, N=trainSample, lastchannelremove = True, labelkey = labelkey)
            # generate validation set
        if valid_image_path is not None:
            export_segmentation_map(valid_image_path, valid_label_path, valid_out_file, image_size=image_size, N=validSample, lastchannelremove = True, labelkey = labelkey)
        # generate test set
        if test_image_path is not None:
            export_segmentation_map(test_image_path,  test_label_path,  test_out_file,  image_size=image_size, N=testSample,  lastchannelremove = True, labelkey = labelkey)


#### Optimize on NeMO_FCN vgg16 based model		
# Input:
#	exporttrainpath: Directory for exported patch images 
# 	exportlabelpath: Directory for exported segmented images
# 	txtfilename: Name of text file to record image names (remember to include '.txt')
#   image_shape: (row x col x channel)
# 	image_size: Size of image patch to use in opt (for symmetric images)
# 	N: Number of images per class (NOTE: because these are segmented maps, the class is only affiliated with the center pixel)
# 	lastchannelremove: Remove last channel or not
# 	labelkey: Naming convention of class labels (NOTE: must be same # as the # of classes)
    
    def NeMO_FCN_vgg16(self, image_shape = image_shape , weight_decay = weight_decay, lr = lr):



        os.environ["CUDA_VISIBLE_DEVICES"] = "0"

        global _SESSION
        config = tf.ConfigProto(allow_soft_placement=True)
        config.gpu_options.allow_growth = True
        _SESSION = tf.Session(config=config)
        K.set_session(_SESSION)


        with open("init_args.yml", 'r') as stream:
            try:
                init_args = yaml.load(stream)
            except yaml.YAMLError as exc:
                print(exc)

        checkpointer = ModelCheckpoint(filepath="./tmp/fcn_vgg16_weights.h5", verbose=1, save_best_only=True)
        lr_reducer = ReduceLROnPlateau(monitor='val_loss',
                                       factor=np.sqrt(0.1),
                                       cooldown=0,
                                       patience=10, min_lr=1e-12)
        early_stopper = EarlyStopping(monitor='val_loss',
                                      min_delta=0.001,
                                      patience=30)
        nan_terminator = TerminateOnNaN()
        SaveWeights = WeightsSaver(filepath='./weights/', N=10)
        #csv_logger = CSVLogger('output/tmp_fcn_vgg16.csv')
             #'output/{}_fcn_vgg16.csv'.format(datetime.datetime.now().isoformat()))

        #check_num = CheckNumericsOps(validation_data=[np.random.random((1, 224, 224, 3)), 1],
        #                             histogram_freq=100)

        # log history during model fit
        csv_logger = CSVLogger('output/log.csv', append=True, separator=';')

        datagen = NeMOImageGenerator(image_shape=[100, 100, 3],
                                            image_resample=True,
                                            pixelwise_center=True,
                                            pixel_mean=[127.5, 127.5, 127.5],
                                            pixelwise_std_normalization=True,
                                            pixel_std=[127.5, 127.5, 127.5])

        train_loader = ImageSetLoader(**init_args['image_set_loader']['train'])
        val_loader = ImageSetLoader(**init_args['image_set_loader']['val'])

        fcn_vgg16 = FCN(input_shape=(100, 100, 3), classes=4, weight_decay=3e-3,
                        weights='imagenet', trainable_encoder=True)
        optimizer = keras.optimizers.Adam(1e-4)

        fcn_vgg16.compile(optimizer=optimizer,
                          loss='categorical_crossentropy',
                          metrics=['accuracy'])

        fcn_vgg16.fit_generator(
            datagen.flow_from_imageset(
                class_mode='categorical',
                classes=4,
                batch_size=10,
                shuffle=True,
                image_set_loader=train_loader),
            steps_per_epoch=80,
            epochs=2,
            validation_data=datagen.flow_from_imageset(
                class_mode='categorical',
                classes=4,
                batch_size=4,
                shuffle=True,
                image_set_loader=val_loader),
            validation_steps=20,
            verbose=1,
            callbacks=[lr_reducer, early_stopper, nan_terminator,checkpointer, csv_logger, SaveWeights])

        #fcn_vgg16.save('output/fcn_vgg16.h5')
