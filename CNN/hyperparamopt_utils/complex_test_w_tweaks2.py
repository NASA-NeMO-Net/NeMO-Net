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
import globalvars
 
 
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
    model = Sequential()
    model.add(Dense(512, input_shape=(784,)))
    model.add(Activation('relu'))
    model.add(Dropout({{uniform(0, 1)}}))
    model.add(Dense({{choice([256, 512, 1024])}}))
    model.add(Activation({{choice(['relu', 'sigmoid'])}}))
    model.add(Dropout({{uniform(0, 1)}}))
 
    
    if conditional({{choice(['three', 'four'])}}) == 'four':
        model.add(Dense(100))
        model.add(Dropout(0.5))
        model.add(Activation('relu'))
 
    model.add(Dense(10))
    model.add(Activation('softmax'))

         
    choiceval = {{choice(['adam', 'sgd', 'rmsprop'])}}
    if choiceval == 'adam':
        adam    = Adam(lr={{choice([10**-6, 10**-5, 10**-4, 10**-3, 10**-2, 10**-1])}}, 
                                decay={{choice([1e-2,1e-3,1e-4])}}, clipnorm=1.)
        optim = adam
    elif choiceval == 'rmsprop':
        rmsprop = RMSprop(lr={{choice([10**-6, 10**-5, 10**-4, 10**-3, 10**-2, 10**-1])}}, 
                                decay={{choice([1e-2,1e-3,1e-4])}}, clipnorm=1.)
        optim = rmsprop
    else:
        sgd     = SGD(lr={{choice([10**-6, 10**-5, 10**-4, 10**-3, 10**-2, 10**-1])}}, 
                                decay={{choice([1e-2,1e-3,1e-4])}}, clipnorm=1.)

        optim = sgd


    model.compile(loss='categorical_crossentropy',
                  optimizer=optim,
                  metrics=['accuracy'])

    globalvars.globalVar += 1

    filepath = "../output/weights_fcn_hyperas" + str(globalvars.globalVar) + ".hdf5"
    checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')

    csv_logger = CSVLogger('../output/hyperas_test_log.csv', 
                                   append=True, separator=';')

    hist = model.fit(X_train, Y_train,
              batch_size={{choice([64, 128])}},
              nb_epoch=1,
              verbose=2,
              validation_data=(X_test, Y_test),
              callbacks=[checkpoint,csv_logger])

    h1   = hist.history
    acc_ = numpy.asarray(h1['acc'])
    loss_ = numpy.asarray(h1['loss'])
    val_loss_ = numpy.asarray(h1['val_loss'])
    val_acc_  = numpy.asarray(h1['val_acc'])
 
    acc_and_loss = numpy.column_stack((acc_, loss_, val_acc_, val_loss_))
    save_file_mlp = '../output/mlp_run_' + '_' + str(globalvars.globalVar) + '.txt'
    with open(save_file_mlp, 'w') as f:
            numpy.savetxt(save_file_mlp, acc_and_loss, delimiter=" ")


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
