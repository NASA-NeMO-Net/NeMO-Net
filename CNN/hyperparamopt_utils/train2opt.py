import os
import yaml
import datetime
import numpy as np
import keras
import keras.backend as K
import tensorflow as tf
import sys
sys.path.append("../utils/") # Adds higher directory to python modules path.
import loadcoraldata_utils as coralutils
#sys.path.append("./tmp/")
from NeMO_models import FCN
from NeMO_generator import NeMOImageGenerator, ImageSetLoader
from keras.callbacks import ReduceLROnPlateau, CSVLogger, EarlyStopping, ModelCheckpoint, TerminateOnNaN
from NeMO_callbacks import CheckNumericsOps, WeightsSaver


# An optimizer method class for training and optimizing models
class TrainOptimizer:
    
    def __init__(self,
                 image_shape    = (100, 100, 3),
                 row_patch_size = 100,
                 col_patch_size = 100,
                 image_size     = 100,
                 num_classes    = 0,
                 weight_decay   = 3e-3,
                 lr             = 1e-4,
                 model          = None,
                 labelkey       = None,
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

        self.image_shape     = tuple(image_shape)
        self.row_patch_size  = row_patch_size
        self.col_patch_size  = col_patch_size
        self.image_size      = image_size
        self.weight_decay    = np.array(weight_decay)
        self.lr              = np.array(lr)
        self.num_classes     = num_classes
        self.train_image_path= train_image_path
        self.train_label_path= train_label_path
        self.train_out_file  = train_out_file
        self.trainSample     = trainSample
        self.valid_image_path= valid_image_path
        self.valid_label_path= valid_label_path
        self.valid_out_file  = valid_out_file
        self.validSample     = validSample
        self.test_image_path = test_image_path
        self.test_label_path = test_label_path
        self.test_out_file   = test_out_file
        self.testSample      = testSample


        
#### generate image patches for train/validation/test
# Input (defined at TrainOptimizer):
# 	train_image_path = folder path to save training images
#   train_label_path = folder path to save labeled images
#   train_out_file   = .txt file that will contain a list of the training images
#   valid_image_path = folder path to save validation images
#   valid_label_path = folder path to save labeled validation images
#   valid_out_file   = .txt file that will contain a list of the valiation images
#   trainSample, validSample, testSample; number of samples per class to generate per each set
    def gen_data(self):
        # generate train set
        if self.train_image_path is not None:
             export_segmentation_map(self.train_image_path, self.train_label_path, self.train_out_file, image_size=self.image_size, N=self.trainSample, lastchannelremove = True, labelkey = self.labelkey)
            # generate validation set
        if self.valid_image_path is not None:
            export_segmentation_map(self.valid_image_path, self.valid_label_path, self.valid_out_file, image_size=self.image_size, N=self.validSample, lastchannelremove = True, labelkey = self.labelkey)
        # generate test set
        if self.test_image_path is not None:
            export_segmentation_map(self.test_image_path,  self.test_label_path,  self.test_out_file,  image_size=self.image_size, N=self.testSample,  lastchannelremove = True, labelkey = self.labelkey)


#### define initial arguments for train/test set generation for model
# Input (defined at TrainOptimizer):
#   'image dir' can be: '../Images/Valid_Patches/'    
#   'label dir' can be: '../Images/ValidRef_Patches/' 
#   'image_set' can be: '../Images/Valid_Patches/NeMO_valid.txt'
#
    def init_args_dict(self):
        
        self.init_args = {'image_set_loader': {'val': {'image_format': 'png', 'image_dir': self.valid_image_path, 
                            'label_format': 'png', 'image_set': self.valid_image_path + self.valid_out_file, 
                            'target_size': [self.row_patch_size, self.col_patch_size], 
                            'label_dir': self.valid_label_path, 'color_mode': 'rgb'}, 
                            'test': {'image_format': 'jpg', 'image_dir': self.test_image_path, 
                            'label_format': 'png',
                            'image_set': self.valid_image_path + self.valid_out_file, 
                            'target_size': [self.row_patch_size, self.col_patch_size], 'label_dir': self.valid_label_path, 
                            'color_mode': 'rgb'}, 
                            'train': {'image_format': 'png', 'image_dir': self.train_image_path, 
                            'label_format': 'png',
                            'image_set': self.train_image_path + self.train_out_file, 
                            'target_size': [self.row_patch_size, self.col_patch_size], 
                            'label_dir': self.train_label_path,
                            'color_mode': 'rgb'}}}
        return self.init_args


#### generate train/test set for model train
# Input (defined at TrainOptimizer):
#
    def gen_train_test_set(self):

        # upload init_args parameters for dataset loader:
        init_args = self.init_args


        return self.
		
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
    
    def NeMO_FCN_vgg16(self):



        os.environ["CUDA_VISIBLE_DEVICES"] = "0"

        global _SESSION
        config = tf.ConfigProto(allow_soft_placement=True)
        config.gpu_options.allow_growth = True
        _SESSION = tf.Session(config=config)
        K.set_session(_SESSION)


        #with open("init_args.yml", 'r') as stream:
        #    try:
        #        init_args = yaml.load(stream)
        #    except yaml.YAMLError as exc:
        #        print(exc)

        # upload init_args parameters for dataset loader:
        init_args = self.init_args

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

        datagen = NeMOImageGenerator(image_shape = self.image_shape,
                                     image_resample=True,
                                     pixelwise_center=True,
                                     pixel_mean=[127.5, 127.5, 127.5],
                                     pixelwise_std_normalization=True,
                                     pixel_std=[127.5, 127.5, 127.5])

        train_loader = ImageSetLoader(**init_args['image_set_loader']['train'])
        val_loader = ImageSetLoader(**init_args['image_set_loader']['val'])

        fcn_vgg16 = FCN(input_shape=self.image_shape, classes=self.num_classes, weight_decay = self.weight_decay,
                        weights='imagenet', trainable_encoder=True)
        optimizer = keras.optimizers.Adam(self.lr)

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
