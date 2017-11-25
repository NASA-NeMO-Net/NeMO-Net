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
from hyperopt import Trials, STATUS_OK, tpe
from hyperas.distributions import choice, uniform, conditional


# An optimizer method class for training and optimizing models
class TrainOptimizer:
    
    def __init__(self,
                 image_shape    = (100, 100, 3),
                 pixel_mean     = (127.5, 127.5, 127.5),
                 pixel_std      = (127.5, 127.5, 127.5),
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
                 num_classes    = 1,
                 batch_size     = [32],
                 weight_decay   = [3e-3],
                 lr             = [1e-4],
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
        self.image_shape     = tuple(image_shape)
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
        #self.validSample     = validSample
        self.test_image_path = test_image_path
        self.test_label_path = test_label_path
        self.test_out_file   = test_out_file
        #self.testSample      = testSample
        self.labelkey        = labelkey
        self.model           = model
        # model params dictionary
        self.num_classes     = num_classes
        # hyperparameters are needed as list in hyperopt/hyperas search
        self.batch_size      = batch_size
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
    def init_args_dict(self, train_image_dir = self.train_image_path,
                             train_label_dir = self.train_label_dir,
                             train_image_set = self.train_image_path + self.train_out_file,
                             valid_image_dir = self.valid_image_path,
                             valid_label_dir = self.valid_label_dir,
                             valid_image_set = self.valid_image_path + self.valid_out_file,
                             test_image_dir  = self.test_image_path,
                             test_label_dir  = self.test_label_dir,
                             test_image_set  = self.test_image_path + self.test_out_file):
        
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
            checkpointer = ModelCheckpoint(filepath="./tmp/" + str(self.model) + "_weights.h5", 
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
            SaveWeights = WeightsSaver(filepath='./weights/', N=self.batch_size[0])
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
        init_args = init_args_dict()

        # generate datasets for train/validation
        datagen = NeMOImageGenerator(image_shape = self.image_shape,
                                     image_resample=True,
                                     pixelwise_center=True,
                                     pixel_mean=self.pixel_mean,
                                     pixelwise_std_normalization=True,
                                     pixel_std=self.pixel_std)

        train_loader = ImageSetLoader(**init_args['image_set_loader']['train'])
        val_loader = ImageSetLoader(**init_args['image_set_loader']['val'])

        return self.datagen, self.train_loader, self.val_loader

#### generate the model with hyperparam tweak
# Input (defined at TrainOptimizer):
#       
    def model2opt(self):

        # define aux params
        steps_per_epoch = list((self.trainSample*self.num_classes)//np.array(self.batch_size))
        val_batch_size  = list(np.array(self.batch_size)//4)
        validation_steps= list((self.validSample*self.num_classes)//np.array(self.val_batch_size))

        # define model to train
        #-----------------------

        if self.model is not None:
            imported_model = self.model(input_shape=self.input_shape, classes=self.num_classes, 
                                    weight_decay = {{choice(self.weight_decay)}},
                                    weights='imagenet', trainable_encoder=True)
            print(imported_model.summary())

            # define optimizers (still TBD for multiple)
            # optimizer = keras.optimizers.Adam(self.lr)
            # adam=keras.optimizers.Adam(lr={{uniform(0,1)}})
            # model.compile(loss='mean_squared_error', optimizer=adam ,metrics=['mean_squared_error'])
            # or:
            #adam = Adam(lr={{uniform(0,1)}})
            #model.compile(loss='binary_crossentropy', metrics=['accuracy'],
            #      optimizer={{choice([adam])}})
            # from hyperopt: 'optimizer' : hp.choice('optimizer',['SGD(lr=0.03, decay=1e-7, momentum=0.15, nesterov=True)','RMSprop','Adadelta','Adam']),

            # imported_model.compile(optimizer={{choice(self.optimizer)}},
            #                         loss={{choice(self.loss)}},
            #                         metrics={{choice(self.metrics)}})

            # using Adam optimizer with multiple lr
            opt=keras.optimizers.Adam(lr={{choice(self.lr)}})
            imported_model.compile(optimizer=opt,
                                    loss={{choice(self.loss)}},
                                    metrics={{choice(self.metrics)}})


            # fit model using fit_generator
            #-------------------------------

            model_callbacks = model_callbacks_list()

            train_generator = self.datagen.flow_from_imageset(
                    class_mode='categorical',
                    classes=self.num_classes,
                    batch_size={{choice(self.batch_size)}},
                    shuffle=True,
                    image_set_loader=self.train_loader)

            validation_generator = self.datagen.flow_from_imageset(
                    class_mode='categorical',
                    classes=self.num_classes,
                    batch_size={{choice(val_batch_size)}},
                    shuffle=True,
                    image_set_loader=self.val_loader)

            imported_model.fit_generator(
                train_generator,
                steps_per_epoch={{choice(steps_per_epoch)}},
                epochs={{choice(self.epochs)}},
                validation_data=validation_generator,
                validation_steps={{choice(validation_steps)}},
                verbose=1,
                callbacks=model_callbacks)

            #using fit_generator to evaluate model
            #-------------------------------------

            score, acc = model.evaluate_generator(generator=validation_generator, 
                                                  steps={{choice(validation_steps)}})

            return {'loss': -acc, 'status': STATUS_OK, 'model': imported_model}


