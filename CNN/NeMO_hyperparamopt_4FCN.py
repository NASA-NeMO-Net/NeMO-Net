from __future__ import print_function
from hyperopt import Trials, STATUS_OK, tpe
from hyperas import optim
from hyperas.distributions import choice, uniform, conditional
from hyperas.utils import eval_hyperopt_space
from keras.models import Sequential
from keras.optimizers import Adam, SGD, RMSprop
from keras.callbacks import ReduceLROnPlateau, CSVLogger, EarlyStopping, ModelCheckpoint, TerminateOnNaN, TensorBoard
from keras.layers.core import Dense, Dropout, Activation
from keras.datasets import mnist
from keras.utils import np_utils
from keras.backend import clear_session
import os
import datetime
import numpy, json
import keras
import keras.backend as K
import tensorflow as tf
import globalvars
import matplotlib.pyplot as plt
from   datetime import time
from   tabulate import tabulate

import sys
sys.path.append("./utils/")
import loadcoraldata_utils as coralutils
sys.path.append("./hyperparamopt_utils/")
from   train2opt import TrainOptimizer
from NeMO_models import FCN
from NeMO_generator import NeMOImageGenerator, ImageSetLoader

 
def data():
    '''
    Data providing function:
 
    This function is separated from model() so that hyperopt
    won't reload data for each evaluation run.
    '''
    
    optModel = TrainOptimizer(labelkey=('Sand', 'Branching', 'Mounding', 'Rock'), train_image_path='../Images/Training_Patches/',
              train_label_path = '../Images/TrainingRef_Patches/', train_out_file = 'NeMO_train.txt',
              valid_image_path = '../Images/Valid_Patches/', valid_label_path = '../Images/ValidRef_Patches/',
              valid_out_file = 'NeMO_valid.txt', pixel_mean = [127.5, 127.5, 127.5], pixel_std = [127.5, 127.5, 127.5],
              num_classes = 4, model = FCN, model_name = "NeMO_FCN", input_shape=(150,150,3), image_size=150)

    train_generator, validation_generator = optModel.gen_data()
    print("train_generator_size: ", train_generator.batch_size)
    print("validation_generator_size: ", validation_generator.batch_size)
    return train_generator, validation_generator

 
def model(train_generator, validation_generator):
    '''
    Model providing function:
 
    Create Keras model with double curly brackets dropped-in as needed.
    Return value has to be a valid python dictionary with two customary keys:
        - loss: Specify a numeric evaluation metric to be minimized
        - status: Just use STATUS_OK and see hyperopt documentation if not feasible
    The last one is optional, though recommended, namely:
        - model: specify the model just created so that we can later use it again.
    '''
    clear_session()

    optModel = TrainOptimizer(labelkey=('Sand', 'Branching', 'Mounding', 'Rock'), train_image_path='../Images/Training_Patches/',
              train_label_path = '../Images/TrainingRef_Patches/', train_out_file = 'NeMO_train.txt',
              valid_image_path = '../Images/Valid_Patches/', valid_label_path = '../Images/ValidRef_Patches/',
              valid_out_file = 'NeMO_valid.txt', pixel_mean = [127.5, 127.5, 127.5], pixel_std = [127.5, 127.5, 127.5],
              num_classes = 4, model = FCN, model_name = "NeMO_FCN", input_shape=(150,150,3), image_size=150)

    model = optModel.model2opt()
         
    choiceval = {{choice(['adam','sgd'])}}
    if choiceval == 'adam':
        adam    = Adam(lr={{choice([10**-6, 10**-5])}}, 
                                decay={{choice([0])}})
        optim = adam
    elif choiceval == 'rmsprop':
        rmsprop = RMSprop(lr={{choice([10**-6, 10**-5, 10**-4, 10**-3, 10**-2])}}, 
                                decay={{choice([1e-2,1e-3,1e-4])}})
        optim = rmsprop
    else:
        sgd     = SGD(lr={{choice([10**-2, 10**-1])}}, 
                                decay={{choice([1e-3,1e-4])}}, 
                                momentum = {{choice([0.1, 0.5,0.9])}} )

        optim = sgd


    model.compile(loss='categorical_crossentropy',
                  optimizer=optim,
                  metrics=['accuracy'])

    globalvars.globalVar += 1

    filepath = './output/weights_' + optModel.model_name + 'hyperas' + str(globalvars.globalVar) + ".hdf5"
    checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')

    csv_logger = CSVLogger('./output/hyperas_' + optModel.model_name + 'test_log.csv', 
                                   append=True, separator=';')
    tensor_board_logfile = './logs/' + optModel.model_name + str(globalvars.globalVar)
    tensor_board = TensorBoard(log_dir=tensor_board_logfile, histogram_freq=0, write_graph=True)


    history = model.fit_generator(
                train_generator,
                steps_per_epoch=80,
                epochs=3,
                validation_data=validation_generator,
                validation_steps=20,
                verbose=0,
                callbacks=[csv_logger])

    

    h1   = history.history
    acc_ = numpy.asarray(h1['acc'])
    loss_ = numpy.asarray(h1['loss'])
    val_loss_ = numpy.asarray(h1['val_loss'])
    val_acc_  = numpy.asarray(h1['val_acc'])
    parameters = space
    opt        = numpy.asarray(parameters["choiceval"])
    if choiceval == 'adam':
      lr         = numpy.asarray(parameters["lr"])
      decay      = numpy.asarray(parameters["decay"])
      momentum   = "none"
    elif choiceval == 'rmsprop':
      lr         = numpy.asarray(parameters["lr_1"])
      decay      = numpy.asarray(parameters["decay_1"])
      momentum   = "none"
    elif choiceval == 'sgd':
      lr         = numpy.asarray(parameters["lr_2"])
      decay      = numpy.asarray(parameters["decay_2"])
      momentum   = numpy.asarray(parameters["momentum"])



    acc_plot = './plots/accuracy_run_' + str(globalvars.globalVar) + ".png"
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('run: ' + str(globalvars.globalVar) + " opt: " + str(opt) + " lr: " + str(lr) + " decay: " + str(decay))
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.savefig(acc_plot)   
    plt.close()  
    
    los_plot = './plots/losses_run_' + str(globalvars.globalVar) + ".png"
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('run: ' + str(globalvars.globalVar) + " opt: " + str(opt) + " lr: " + str(lr) + " decay: " + str(decay))
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.savefig(los_plot)   
    plt.close()  
    

    
    print("parameters for run " + str(globalvars.globalVar) + ":")
    print("-------------------------------")
    print(parameters)
    print("opt: ", opt)
    print("lr: ", lr)
    print("decay: ", decay)
    print("momentum: ", momentum)
    print("val_accuracy: ",val_acc_)
 
    acc_and_loss = numpy.column_stack((acc_, loss_, val_acc_, val_loss_))
    save_file_model = './output/' + optModel.model_name + '_run_' + '_' + str(globalvars.globalVar) + '.txt'
    with open(save_file_model, 'w') as f:
            numpy.savetxt(save_file_model, acc_and_loss, delimiter=",")


    score, acc = model.evaluate_generator(generator=validation_generator, 
                                                  steps=20)
    print('Test accuracy:', acc)

    save_file_params = './output/params_run_' + '_' + str(globalvars.globalVar) + '.txt'
    rownames  = numpy.array(['Run', 'optimizer', 'learning_rate', 'decay', 'train_accuracy','train_loss','val_accuracy', 'val_loss', 'test_accuracy'])
    rowvals   = (str(globalvars.globalVar), opt, lr, decay, acc_[-1], loss_[-1], val_acc_[-1], val_loss_[-1],acc)

    DAT =  numpy.column_stack((rownames, rowvals))
    numpy.savetxt(save_file_params, DAT, delimiter=",",fmt="%s")

    return {'loss': -acc, 'status': STATUS_OK, 'model': model}
 
if __name__ == '__main__':

    
    optModel = TrainOptimizer(labelkey=('Sand', 'Branching', 'Mounding', 'Rock'), train_image_path='../Images/Training_Patches/',
              train_label_path = '../Images/TrainingRef_Patches/', train_out_file = 'NeMO_train.txt',
              valid_image_path = '../Images/Valid_Patches/', valid_label_path = '../Images/ValidRef_Patches/',
              valid_out_file = 'NeMO_valid.txt', pixel_mean = [127.5, 127.5, 127.5], pixel_std = [127.5, 127.5, 127.5],
              num_classes = 4, model = FCN, model_name = "NeMO_FCN", input_shape=(150,150,3), image_size=150)

    trials=Trials()

    train_generator, validation_generator = data()

    best_run, best_model, space = optim.minimize(model=model,
                                          data=data,
                                          algo=tpe.suggest,
                                          max_evals=2,
                                          trials=trials,
                                          eval_space=True,
                                          return_space=True)
  
    print("validation_generator_size: ", validation_generator.batch_size)
    print("Evalutation of best performing model:")
    print("Parameters of best run", best_run)
    #print(best_model.evaluate_generator(generator=validation_generator, steps=20))
    print(best_model.evaluate(validation_generator))
    json.dump(best_run, open('./output/best_run' + optModel.model_name + '.txt', 'w'))

    for t, trial in enumerate(trials):
        vals = trial.get('misc').get('vals')
        print("Trial %s vals: %s" % (t, vals))
        print(eval_hyperopt_space(space, vals))
