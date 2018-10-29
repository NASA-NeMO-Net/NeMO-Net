from __future__ import print_function
from hyperopt import Trials, STATUS_OK, tpe
from hyperas import optim
from hyperas.distributions import choice, uniform, conditional
from hyperas.utils import eval_hyperopt_space
from keras.models import Sequential
from keras.optimizers import Adam, SGD, RMSprop
from keras.callbacks import ReduceLROnPlateau, CSVLogger, EarlyStopping, ModelCheckpoint, TerminateOnNaN
from keras.layers.core import Dense, Dropout, Activation
from keras.datasets import mnist
from keras.utils import np_utils
import numpy, json
import sys
sys.path.append("./utils/") # Adds higher directory to python modules path.
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

 
 
def data():
    '''
    Data providing function:
 
    This function is separated from model() so that hyperopt
    won't reload data for each evaluation run.
    '''
    (X_train, y_train), (X_test, y_test) = mnist.load_data()
    X_train = X_train.reshape(60000, 784)
    X_test = X_test.reshape(10000, 784)
    X_train = X_train.astype('float32')
    X_test = X_test.astype('float32')
    X_train /= 255
    X_test /= 255
    nb_classes = 10
    Y_train = np_utils.to_categorical(y_train, nb_classes)
    Y_test = np_utils.to_categorical(y_test, nb_classes)
    return X_train, Y_train, X_test, Y_test
 
 
def model(X_train, Y_train, X_test, Y_test):
    '''
    Model providing function:
 
    Create Keras model with double curly brackets dropped-in as needed.
    Return value has to be a valid python dictionary with two customary keys:
        - loss: Specify a numeric evaluation metric to be minimized
        - status: Just use STATUS_OK and see hyperopt documentation if not feasible
    The last one is optional, though recommended, namely:
        - model: specify the model just created so that we can later use it again.
    '''
    imported_model = FCN(input_shape=(100, 100, 3), classes=4, 
                                    weight_decay = 3e-3,
                                    weights='imagenet', trainable_encoder=True)
    print(imported_model.summary())

    opt = Adam(lr={{choice([0.01, 0.001, 0.0001,0.00001])}}, decay={{choice([1e-2,1e-3,1e-4])}})
    imported_model.compile(optimizer=opt,
                                    loss='categorical_crossentropy',
                                    metrics=['accuracy'])

    globalvars.globalVar += 1

    csv_logger = CSVLogger('../output/hyperas_fcn_log.csv', 
                                   append=True, separator=';')
    filepath = "weights_fcn_hyperas" + str(globalvars.globalVar) + ".hdf5"
    checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')

    model.fit(X_train, Y_train,
              batch_size={{choice([64, 128])}},
              nb_epoch=1,
              verbose=2,
              validation_data=(X_test, Y_test),
              callbacks=[csv_logger,checkpoint])
    score, acc = model.evaluate(X_test, Y_test, verbose=0)
    print('Test accuracy:', acc)
    return {'loss': -acc, 'status': STATUS_OK, 'model': model}
 
if __name__ == '__main__':
    best_run, best_model, space = optim.minimize(model=model,
                                          data=data,
                                          algo=tpe.suggest,
                                          max_evals=3,
                                          trials=Trials(),
                                          eval_space=True,
                                          return_space=True)
    X_train, Y_train, X_test, Y_test = data()
    print("Evalutation of best performing model:")
    print("Parameters of best run", best_run)
    #real_param_values = eval_hyperopt_space(space, best_run)
    #print("Parameters of best run", real_param_values)
    print(best_model.evaluate(X_test, Y_test))
    json.dump(best_run, open("../output/best_run.txt", 'w'))
