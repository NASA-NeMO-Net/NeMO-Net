#coding=utf-8

from __future__ import print_function

try:
    from hyperopt import Trials, STATUS_OK, tpe
except:
    pass

try:
    from hyperas import optim
except:
    pass

try:
    from hyperas.distributions import choice, uniform, conditional
except:
    pass

try:
    from hyperas.utils import eval_hyperopt_space
except:
    pass

try:
    from keras.models import Sequential
except:
    pass

try:
    from keras.optimizers import Adam, SGD, RMSprop
except:
    pass

try:
    from keras.callbacks import ReduceLROnPlateau, CSVLogger, EarlyStopping, ModelCheckpoint, TerminateOnNaN, TensorBoard
except:
    pass

try:
    from keras.layers.core import Dense, Dropout, Activation
except:
    pass

try:
    from keras.datasets import mnist
except:
    pass

try:
    from keras.utils import np_utils
except:
    pass

try:
    import numpy, json
except:
    pass

try:
    import globalvars
except:
    pass

try:
    import matplotlib.pyplot as plt
except:
    pass

try:
    from datetime import time
except:
    pass

try:
    from tabulate import tabulate
except:
    pass

try:
    from train2opt import TrainOptimizer
except:
    pass

try:
    from NeMO_models import FCN
except:
    pass

try:
    from NeMO_generator import NeMOImageGenerator, ImageSetLoader
except:
    pass
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials
from hyperas.distributions import conditional

'''
Data providing function:

This function is separated from model() so that hyperopt
won't reload data for each evaluation run.
'''
optModel = TrainOptimizer(labelkey=('Sand', 'Branching', 'Mounding', 'Rock'), train_image_path='../Images/Training_Patches/',
          train_label_path = '../Images/TrainingRef_Patches/', train_out_file = 'NeMO_train.txt',
          valid_image_path = '../Images/Valid_Patches/', valid_label_path = '../Images/ValidRef_Patches/',
          valid_out_file = 'NeMO_valid.txt', pixel_mean = [127.5, 127.5, 127.5], pixel_std = [127.5, 127.5, 127.5],
          num_classes = 4, model = FCN, model_name = "NeMO_FCN")

train_generator, validation_generator = optModel.gen_data()



def keras_fmin_fnct(space):

    '''
    Model providing function:
 
    Create Keras model with double curly brackets dropped-in as needed.
    Return value has to be a valid python dictionary with two customary keys:
        - loss: Specify a numeric evaluation metric to be minimized
        - status: Just use STATUS_OK and see hyperopt documentation if not feasible
    The last one is optional, though recommended, namely:
        - model: specify the model just created so that we can later use it again.
    '''
    
    optModel = TrainOptimizer(labelkey=('Sand', 'Branching', 'Mounding', 'Rock'), train_image_path='../Images/Training_Patches/',
              train_label_path = '../Images/TrainingRef_Patches/', train_out_file = 'NeMO_train.txt',
              valid_image_path = '../Images/Valid_Patches/', valid_label_path = '../Images/ValidRef_Patches/',
              valid_out_file = 'NeMO_valid.txt', pixel_mean = [127.5, 127.5, 127.5], pixel_std = [127.5, 127.5, 127.5],
              num_classes = 4, model = FCN, model_name = "NeMO_FCN")

    model = optModel.model2opt()
         
    choiceval = space['choiceval']
    if choiceval == 'adam':
        adam    = Adam(lr=space['lr'], 
                                decay=space['decay'], clipnorm=1.)
        optim = adam
    elif choiceval == 'rmsprop':
        rmsprop = RMSprop(lr=space['lr_1'], 
                                decay=space['decay_1'], clipnorm=1.)
        optim = rmsprop
    else:
        sgd     = SGD(lr=space['lr_2'], 
                                decay=space['decay_2'], clipnorm=1.)

        optim = sgd


    model.compile(loss='categorical_crossentropy',
                  optimizer=optim,
                  metrics=['accuracy'])

    globalvars.globalVar += 1

    filepath = '../output/weights_' + optModel.model_name + 'hyperas' + str(globalvars.globalVar) + ".hdf5"
    checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')

    csv_logger = CSVLogger('../output/hyperas_' + optModel.model_name + 'test_log.csv', 
                                   append=True, separator=';')
    tensor_board_logfile = '../logs/' + optModel.model_name + str(globalvars.globalVar)
    tensor_board = TensorBoard(log_dir=tensor_board_logfile, histogram_freq=0, write_graph=True)


    if 'results' not in globals():
        global results
        results = []

    
    steps_per_epoch = (optModel.trainSample*optModel.num_classes)//np.array(optModel.batch_size)

    history = model.fit_generator(
                train_generator,
                steps_per_epoch=steps_per_epoch,
                epochs=3,
                validation_data=validation_generator,
                validation_steps=steps_per_epoch,
                verbose=2,
                callbacks=[checkpoint,csv_logger,tensor_board])



    print(history.history.keys())
    

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
    elif choiceval == 'rmsprop':
      lr         = numpy.asarray(parameters["lr_1"])
      decay      = numpy.asarray(parameters["decay_1"])
    elif choiceval == 'sgd':
      lr         = numpy.asarray(parameters["lr_2"])
      decay      = numpy.asarray(parameters["decay_2"])

    results.append(parameters)


    acc_plot = '../plots/accuracy_run_' + str(globalvars.globalVar) + ".png"
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('run: ' + str(globalvars.globalVar) + " opt: " + str(opt) + " lr: " + str(lr) + " decay: " + str(decay))
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.savefig(acc_plot)   
    plt.close()  
    
    los_plot = '../plots/losses_run_' + str(globalvars.globalVar) + ".png"
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
    print("val_accuracy: ",val_acc_)
 
    acc_and_loss = numpy.column_stack((acc_, loss_, val_acc_, val_loss_))
    save_file_model = '../output/' + optModel.model_name + '_run_' + '_' + str(globalvars.globalVar) + '.txt'
    with open(save_file_model, 'w') as f:
            numpy.savetxt(save_file_model, acc_and_loss, delimiter=",")


    score, acc = model.evaluate_generator(generator=validation_generator, 
                                                  steps=steps_per_epoch, verbose=0)
    print('Test accuracy:', acc)

    save_file_params = '../output/params_run_' + '_' + str(globalvars.globalVar) + '.txt'
    rownames  = numpy.array(['Run', 'optimizer', 'learning_rate', 'decay', 'train_accuracy','train_loss','val_accuracy', 'val_loss', 'test_accuracy'])
    rowvals   = (str(globalvars.globalVar), opt, lr, decay, acc_[-1], loss_[-1], val_acc_[-1], val_loss_[-1],acc)

    DAT =  numpy.column_stack((rownames, rowvals))
    numpy.savetxt(save_file_params, DAT, delimiter=",",fmt="%s")

    return {'loss': -acc, 'status': STATUS_OK, 'model': model}

def get_space():
    return {
        'choiceval': hp.choice('choiceval', ['adam','rmsprop','sgd']),
        'lr': hp.uniform('lr', 10**-6, 10**-1),
        'decay': hp.uniform('decay', 1e-4,1e-1),
        'lr_1': hp.choice('lr_1', [10**-6, 10**-5, 10**-4, 10**-3, 10**-2, 10**-1]),
        'decay_1': hp.choice('decay_1', [1e-2,1e-3,1e-4]),
        'lr_2': hp.choice('lr_2', [10**-6, 10**-5, 10**-4, 10**-3, 10**-2, 10**-1]),
        'decay_2': hp.choice('decay_2', [1e-2,1e-3,1e-4]),
    }
