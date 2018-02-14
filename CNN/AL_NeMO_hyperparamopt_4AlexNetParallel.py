from __future__ import print_function
from hyperopt import Trials, STATUS_OK, tpe
from hyperas import optim
from hyperas.distributions import choice, uniform, conditional
from hyperas.utils import eval_hyperopt_space
from keras.models import Sequential
from keras.optimizers import Adam, SGD, RMSprop
from keras.callbacks import ReduceLROnPlateau, CSVLogger, EarlyStopping, ModelCheckpoint, TerminateOnNaN, TensorBoard
from keras.layers.core import Dense, Dropout, Activation
from keras.layers import Input
from keras.models import Model
from keras.utils import np_utils
from keras.backend import clear_session
import keras.backend as K
import os
import datetime
import numpy as np
import json
import keras
import keras.backend as K
import tensorflow as tf
import globalvars
import yaml
import matplotlib.pyplot as plt
from   datetime import time
from   tabulate import tabulate

import sys
sys.path.append("./utils/")
import loadcoraldata_utils as coralutils
sys.path.append("./hyperparamopt_utils/")
from   train2opt import TrainOptimizer
from NeMO_models import FCN, AlexNet, Alex_Hyperopt_ParallelNet
from NeMO_encoders import Alex_Encoder, Alex_Hyperopt_Encoder
from NeMO_generator import NeMOImageGenerator, ImageSetLoader
from NeMO_callbacks import CheckNumericsOps, WeightsSaver
 
def data():
  imgpath = '../Images/BTPB-WV2-2012-15-8Band-mosaic-GeoTiff-Sample-AOI/BTPB-WV2-2012-15-8Band-mosaic-GeoTiff-Sample-AOI.tif'
  tfwpath = '../Images/BTPB-WV2-2012-15-8Band-mosaic-GeoTiff-Sample-AOI/BTPB-WV2-2012-15-8Band-mosaic-GeoTiff-Sample-AOI.tfw'
  truthpath = '../Images/BIOT-PerosBanhos-sample-habitat-map/BIOT-PerosBanhos-sample-habitat-map.shp'
  PerosBanhos = coralutils.CoralData(imgpath, Truthpath=truthpath, load_type="raster", tfwpath=tfwpath)
  labelkey = PerosBanhos.class_labels

  # labelkey = ['Sand', 'Branching', 'Mounding', 'Rock']

  num_classes = len(labelkey)
  batch_size = 120
  model_name = "NeMO_AlexNetParallel"

  with open("init_args - AlexNetParallel_Raster.yml", 'r') as stream:
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
  pixel_mean =1023.5*np.ones(num_channels)
  pixel_std = 1023.5*np.ones(num_channels)

  # generate datasets for train/validation
  datagen = NeMOImageGenerator(image_shape = (y,x,num_channels),
    image_resample=True, pixelwise_center=True,
    pixel_mean=pixel_mean, pixelwise_std_normalization=True,
    pixel_std=pixel_std)

  train_generator = datagen.flow_from_NeMOdirectory(train_loader.image_dir,
    target_size=(x,y),
    color_mode=train_loader.color_mode,
    classes = labelkey,
    class_mode = 'categorical',
    batch_size = batch_size,
    shuffle=True)

  validation_generator = datagen.flow_from_NeMOdirectory(val_loader.image_dir,
    target_size=(x,y),
    color_mode=val_loader.color_mode,
    classes = labelkey,
    class_mode = 'categorical',
    batch_size = batch_size,
    shuffle=True)

  print("train_generator_size: ", train_generator.batch_size)
  print("validation_generator_size: ", validation_generator.batch_size)
  return train_generator, validation_generator, model_name, num_channels

 
def model(train_generator, validation_generator, model_name, num_channels):
  clear_session()
  num_classes = train_generator.num_class
  print("==============================================================================")

  conv_layers = {{choice([3])}}
  full_layers = 1
  conv_params= {"filters": [64, 128, 256],
    "conv_size": [(7,7),(3,3),(3,3)],
    "pool_size": [(2,2),(2,2),(1,1)],
    "dilation_rate": [(1,1),(1,1),(1,1)],
    "batchnorm_bool": [True,True,True],
    "full_filters": [2048],
    "dropout": [0.5]}
  crop_shapes = [(25,25),(50,50),(100,100)]

  model = Alex_Hyperopt_ParallelNet(input_shape=train_generator.image_shape, crop_shapes=crop_shapes,
    classes=num_classes, conv_layers=conv_layers, full_layers=full_layers)
  model.summary()
  optim = Adam(lr=10**-5, decay=0)
      
  model.compile(loss='categorical_crossentropy',
                optimizer=optim,
                metrics=['accuracy'])

  globalvars.globalVar += 1

  filepath = './output/weights_' + model_name + '_hyperas' + str(globalvars.globalVar) + ".h5"
  early_stopper = EarlyStopping(monitor='val_loss', min_delta=0.001, patience=30)
  checkpointer = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
  lr_reducer = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=0.001)
  nan_terminator = TerminateOnNaN()

  csv_logger = CSVLogger('./output/hyperas_' + model_name + '_log.csv', 
                                 append=True, separator=';')
  tensor_board_logfile = './logs/' + model_name + str(globalvars.globalVar)
  tensor_board = TensorBoard(log_dir=tensor_board_logfile, histogram_freq=0, write_graph=True)
  SaveWeights = WeightsSaver(filepath='./weights/', model_name=model_name, N=50)

  history = model.fit_generator(
              train_generator,
              steps_per_epoch=200,
              epochs=100,
              validation_data=validation_generator,
              validation_steps=10,
              verbose=1,
              callbacks=[lr_reducer, early_stopper, csv_logger, checkpointer, SaveWeights])

  
  h1   = history.history
  acc_ = np.asarray(h1['acc'])
  loss_ = np.asarray(h1['loss'])
  val_loss_ = np.asarray(h1['val_loss'])
  val_acc_  = np.asarray(h1['val_acc'])
  parameters = space
  print("Hyperas Parameters:")
  print(parameters)


  acc_plot = './plots/accuracy_run_' + model_name + '_' + str(globalvars.globalVar) + ".png"
  plt.plot(history.history['acc'])
  plt.plot(history.history['val_acc'])
  plt.title('run: ' + str(globalvars.globalVar))
  plt.ylabel('accuracy')
  plt.xlabel('epoch')
  plt.legend(['train', 'test'], loc='upper left')
  plt.savefig(acc_plot)
  plt.close()  
  
  los_plot = './plots/losses_run_' + model_name + '_' + str(globalvars.globalVar) + ".png"
  plt.plot(history.history['loss'])
  plt.plot(history.history['val_loss'])
  plt.title('run: ' + str(globalvars.globalVar))
  plt.ylabel('loss')
  plt.xlabel('epoch')
  plt.legend(['train', 'test'], loc='upper left')
  plt.savefig(los_plot)   
  plt.close()  
  

  acc_and_loss = np.column_stack((acc_, loss_, val_acc_, val_loss_))
  save_file_model = './output/' + model_name + '_run_' + '_' + str(globalvars.globalVar) + '.txt'
  with open(save_file_model, 'w') as f:
    np.savetxt(save_file_model, acc_and_loss, delimiter=",")


  score, acc = model.evaluate_generator(generator=validation_generator, 
                                                steps=10)
  print('Test accuracy:', acc)

  # save_file_params = './output/params_run_' + '_' + str(globalvars.globalVar) + '.txt'
  # rownames  = np.array(['Run', 'optimizer', 'learning_rate', 'decay', 'train_accuracy','train_loss','val_accuracy', 'val_loss', 'test_accuracy'])
  # rowvals   = (str(globalvars.globalVar), opt, lr, decay, acc_[-1], loss_[-1], val_acc_[-1], val_loss_[-1],acc)

  # DAT =  np.column_stack((rownames, rowvals))
  # np.savetxt(save_file_params, DAT, delimiter=",",fmt="%s")

  filename = './output/temp_saveacc.txt'
  if globalvars.globalVar == 1:
    f = open(filename,"w")
  else:
    f = open(filename,"a")
  f.write("%.6f \n" %acc)
  f.close()

  return {'loss': -acc, 'status': STATUS_OK, 'model': model}
 
if __name__ == '__main__':

  os.environ["CUDA_VISIBLE_DEVICES"] = "0"

  global _SESSION
  config = tf.ConfigProto(allow_soft_placement=False)
  config.gpu_options.allow_growth = False
  _SESSION = tf.Session(config=config)
  K.set_session(_SESSION)

  trials=Trials()
  train_generator, validation_generator, model_name, num_channels = data()

  best_run, best_model, space = optim.minimize(model=model,
                                        data=data,
                                        algo=tpe.suggest,
                                        max_evals=1,
                                        trials=trials,
                                        eval_space=True,
                                        return_space=True)

  # print("validation_generator_size: ", validation_generator.batch_size)
  print("=============================================================================================================")
  print("SUMMARY:")

  print("Evalutation of best performing model:")
  print("Parameters of best run", best_run)
  # print(best_model.evaluate_generator(generator=validation_generator, steps=5))
  # print(best_model.evaluate(validation_generator))
  json.dump(best_run, open('./output/best_run_' + model_name + '.txt', 'w'))

  f = open("./output/temp_saveacc.txt","r")

  for t, trial in enumerate(trials):
      temp_acc = f.readline()
      print("========================================================================")
      vals = trial.get('misc').get('vals')
      # print("Trial %s vals: %s" % (t, vals))
      print("Trial %s:" %(t+1))
      print(eval_hyperopt_space(space, vals))
      print("Accuracy: ", temp_acc)
  f.close()
