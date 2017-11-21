# import hyperparam optimization scheme
from __future__ import print_function
from hyperopt import Trials, STATUS_OK, tpe
from hyperas import optim
from hyperas.distributions import choice, uniform
# import other moduls
import os
import yaml
import datetime
import numpy as np
import keras
import keras.backend as K
import tensorflow as tf
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


# define GPU environment
#-----------------------
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

global _SESSION
config = tf.ConfigProto(allow_soft_placement=True)
config.gpu_options.allow_growth = True
_SESSION = tf.Session(config=config)
K.set_session(_SESSION)


# generate run data
#------------------


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
    
    visualization_mnist(X_test)
    
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
    model = Sequential()
    model.add(Dense(512, input_shape=(784,)))
    model.add(Activation('relu'))
    model.add(Dropout({{uniform(0, 1)}}))
    model.add(Dense({{choice([256, 512])}}))
    model.add(Activation('relu'))
    model.add(Dropout({{uniform(0, 1)}}))
    model.add(Dense(10))
    model.add(Activation('softmax'))

    optimizer = {{choice(['rmsprop', 'adam', 'sgd'])}}
    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])

    model.fit(X_train, Y_train,
              batch_size={{choice([64, 128])}},
              nb_epoch=1,
              verbose=2,
              validation_data=(X_test, Y_test))
    score, acc = model.evaluate(X_test, Y_test, verbose=0)
    print('Test accuracy:', acc)
    return {'loss': -acc, 'status': STATUS_OK, 'model': model}

if __name__ == '__main__':

    X_train, Y_train, X_test, Y_test = data()
    functions=[visualization_mnist]
    best_run, best_model = optim.minimize(model=model,
                                          data=data,
                                          functions=functions,
                                          algo=tpe.suggest,
                                          max_evals=5,
                                          trials=Trials())

    print("Evalutation of best performing model:")
    print(best_model.evaluate(X_test, Y_test))
