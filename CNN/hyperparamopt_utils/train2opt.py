import os
import yaml
import pickle
import time
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
from hyperopt import Trials, STATUS_OK, tpe, hp
from hyperas.distributions import choice, uniform, conditional


# An optimizer method class for training and optimizing models
class TrainOptimizer:
    
    def __init__(self,
                 input_shape    = (100, 100, 3),
                 pixel_mean     = (0.,0.,0.),
                 pixel_std      = (1.,1.,1.),
                 row_patch_size = 100,
                 col_patch_size = 100,
                 image_size     = 100,
                 labelkey       = None,
                 train_image_path=None,
                 train_label_path=None,
                 train_out_file = None,
                 trainSample    = 200,
                 valid_image_path=None,
                 valid_label_path=None,
                 valid_out_file = None,
                 test_image_path= None,
                 test_label_path= None,
                 test_out_file  = None,
                 model          = None,
                 model_name     = None,
                 num_classes    = 1,
                 batch_size     = 10,
                 weight_decay   = 3e-3,
                 lr             = 1e-4,
                 optimizer      = ['adam'],
                 loss           = ['categorical_crossentropy'],
                 metrics        = ['accuracy'],
                 checkpoint     = True,
                 lr_reduce      = True,
                 early_stop     = True,
                 nan_terminate  = True,
                 save_weights   = True,
                 csv_log        = True):

        # data set param dictionary
        self.input_shape     = tuple(input_shape)
        self.row_patch_size  = row_patch_size
        self.col_patch_size  = col_patch_size
        self.image_size      = image_size
        self.pixel_mean      = np.array(pixel_mean)
        self.pixel_std       = np.array(pixel_std)
        self.train_image_path= train_image_path
        self.train_label_path= train_label_path
        self.train_out_file  = train_out_file
        self.trainSample     = trainSample
        self.valid_image_path= valid_image_path
        self.valid_label_path= valid_label_path
        self.valid_out_file  = valid_out_file
        #self.validSample     = trainSample//10
        self.test_image_path = test_image_path
        self.test_label_path = test_label_path
        self.test_out_file   = test_out_file
        #self.testSample      = trainSample//10
        self.labelkey        = labelkey
        self.model           = model # this is model function
        self.model_name      = model_name # this is model name string
        # model params dictionary
        self.num_classes     = num_classes
        # hyperparameters are needed as list in hyperopt/hyperas search
        self.batch_size      = batch_size
        #self.val_batch_size  = list(np.array(self.batch_size)//4)
        self.weight_decay    = weight_decay # this is for regularization of weights
        self.lr              = lr
        self.optimizer       = optimizer # this version only tweals lr within one optimizer
        self.loss            = loss # this can also be: 'mean_squared_error'
        self.metrics         = metrics # this can also be: 'mean_squared_error'
        # callbacks list
        self.checkpoint      = checkpoint
        self.lr_reduce       = lr_reduce
        self.early_stop      = early_stop
        self.nan_terminate   = nan_terminate
        self.save_weights    = save_weights
        self.csv_log         = csv_log
        


        
#### generate image patches for train/validation/test
# Input (defined at TrainOptimizer):
# 	train_image_path = folder path to save training images
#   train_label_path = folder path to save labeled images
#   train_out_file   = .txt file that will contain a list of the training images
#   valid_image_path = folder path to save validation images
#   valid_label_path = folder path to save labeled validation images
#   valid_out_file   = .txt file that will contain a list of the valiation images
#   trainSample, validSample, testSample; number of samples per class to generate per each set
#   labelkey: Naming convention of class labels (NOTE: must be same # as the # of classes) (string names of classes)
    def gen_img_set(self):
        # generate train set
        if self.train_image_path is not None:
             coralutils.export_segmentation_map(self.train_image_path, self.train_label_path, self.train_out_file, 
                                                image_size=self.image_size, N=self.trainSample, lastchannelremove = True, labelkey = self.labelkey)
            # generate validation set
        if self.valid_image_path is not None:
            coralutils.export_segmentation_map(self.valid_image_path, self.valid_label_path, self.valid_out_file, 
                                               image_size=self.image_size, N=self.trainSample//10, lastchannelremove = True, labelkey = self.labelkey)
        # generate test set
        if self.test_image_path is not None:
            coralutils.export_segmentation_map(self.test_image_path,  self.test_label_path,  self.test_out_file,  
                                               image_size=self.image_size, N=self.trainSample//10,  lastchannelremove = True, labelkey = self.labelkey)


#### define initial arguments dictionary for train/test data set generation for model
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


#### define model callbacks list
# Input (defined at TrainOptimizer):
#
    def model_callbacks_list(self):
        
        self.model_callbacks = []

        if self.checkpoint is True:
            print(self.model_name)
            checkpointer = ModelCheckpoint(filepath="./tmp/" + self.model_name + "_weights.h5", 
                                           verbose=1, save_best_only=True)
            self.model_callbacks.append(checkpointer)

        if self.lr_reduce is True:
            lr_reducer = ReduceLROnPlateau(monitor='val_loss',
                                           factor=np.sqrt(0.1),
                                           cooldown=0,
                                           patience=10, min_lr=1e-12)
            self.model_callbacks.append(lr_reducer)
        
        if self.early_stop is True:
            early_stopper = EarlyStopping(monitor='val_loss',
                                          min_delta=0.001,
                                          patience=30)
            self.model_callbacks.append(early_stopper)

        if self.nan_terminate is True:
            nan_terminator = TerminateOnNaN()
            self.model_callbacks.append(nan_terminator)

        if self.save_weights is True:
            SaveWeights = WeightsSaver(filepath='./weights/', N=10)
            self.model_callbacks.append(SaveWeights)
        
        # log history during model fit
        if self.csv_log is True:
            csv_logger = CSVLogger('output/log.csv', 
                                   append=True, separator=';')
            self.model_callbacks.append(csv_logger)

        return self.model_callbacks



#### generate train/test set for model train
# Input (defined at TrainOptimizer):
# this is an implementation of imageset Loader, for parallel running
#
    def gen_data(self):

        # upload init_args parameters for dataset loader:
        init_args  = self.init_args_dict()

        batch_size = self.batch_size

        # generate datasets for train/validation
        datagen = NeMOImageGenerator(image_shape = self.input_shape,
                                     image_resample=True,
                                     pixelwise_center=True,
                                     pixel_mean=self.pixel_mean,
                                     pixelwise_std_normalization=True,
                                     pixel_std=self.pixel_std)

        train_loader = ImageSetLoader(**init_args['image_set_loader']['train'])
        val_loader = ImageSetLoader(**init_args['image_set_loader']['val'])


        # generate model data

        self.train_generator = datagen.flow_from_imageset(
                    class_mode='categorical',
                    classes=self.num_classes,
                    batch_size=batch_size,
                    shuffle=True,
                    image_set_loader=train_loader)

        self.validation_generator = datagen.flow_from_imageset(
                    class_mode='categorical',
                    classes=self.num_classes,
                    batch_size=batch_size,
                    shuffle=True,
                    image_set_loader=val_loader)


        return self.train_generator, self.validation_generator

#### generate the model with hyperparam space using hyperopt dict method
# Input (defined at TrainOptimizer):
#       
    def gen_param_space(self):

    # space = {'choice': hp.choice('num_layers',
    #                 [ {'layers':'two', },
    #                 {'layers':'three',
    #                 'units3': hp.uniform('units3', 64,1024), 
    #                 'dropout3': hp.uniform('dropout3', .25,.75)}
    #                 ]),

    #         'units1': hp.uniform('units1', 64,1024),
    #         'units2': hp.uniform('units2', 64,1024),

    #         'dropout1': hp.uniform('dropout1', .25,.75),
    #         'dropout2': hp.uniform('dropout2',  .25,.75),

    #         'batch_size' : hp.uniform('batch_size', 28,128),

    #         'nb_epochs' :  100,
    #         'optimizer': hp.choice('optimizer',['adadelta','adam','rmsprop']),
    #         'activation': 'relu'
    #     }
        # space = hp.choice('a',
        # [
        #     ('case 1', 1 + hp.lognormal('c1', 0, 1)),
        #     ('case 2', hp.uniform('c2', -10, 10))
        # ])

        param_space = {

                'batch_size'   : hp.choice('batch_size', [16,32]),
                'epochs'       : hp.choice('epochs', [2,10]),
                'lr'           : hp.choice('lr',[0.01, 0.001, 0.0001])
            }
        # need to add steps_per_epoch based on number of training samples and batch size
        return param_space


#### generate the model with hyperparam space as params
# Input (params defined by space):
#    to run from script: test_model = optModel.model2opt(param_space) (optModel is a class of TrainOptimizer)
#   
    def model2opt(self):

        #print ('Params testing: ', param_space)
        # define aux params
        #steps_per_epoch = (self.trainSample*self.num_classes)//np.array(self.batch_size)
        #self.val_batch_size  = list(np.array(self.batch_size)//4)
        #print("val_batch_size",self.val_batch_size)
        #validation_steps= list(((self.trainSample//10)*self.num_classes)//np.array(self.val_batch_size))

        # define model to train
        #-----------------------

        if self.model is not None:
            self.imported_model = self.model(input_shape=self.input_shape, classes=self.num_classes, 
                                    weight_decay = 3e-3,
                                    weights='imagenet', trainable_encoder=True)
            #print(self.imported_model.summary())

            
            # # define callbacks
            # model_callbacks = model_callbacks_list()

            # # fit model
            # imported_model.fit_generator(
            #     train_generator,
            #     steps_per_epoch=steps_per_epoch,
            #     epochs=param_space['epochs'],
            #     validation_data=validation_generator,
            #     validation_steps=steps_per_epoch,
            #     verbose=1,
            #     callbacks=model_callbacks)

            # #using fit_generator to evaluate model
            # #-------------------------------------

            # score, acc = imported_model.evaluate_generator(generator=validation_generator, 
            #                                                steps=steps_per_epoch)

            return  self.imported_model #{'loss': -acc, 'status': STATUS_OK, 'model': imported_model}


