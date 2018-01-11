# hyperas example with datagen (fkeras fit_generator)

from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras.callbacks import ModelCheckpoint
from keras import backend as K
from hyperas import optim
from hyperopt import Trials, STATUS_OK, tpe
from hyperas.distributions import choice, uniform, conditional
import h5py

def model(train_generator, validation_generator):
    img_width, img_height = 250, 1500
    input_shape = (img_width, img_height, 3)
    nb_train_samples = 2000
    nb_validation_samples = 800

    epochs = 2
    batch_size = 256
    model = Sequential()
    model.add(Conv2D(32, (3, 3), input_shape=input_shape))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(32, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D({{choice([32,64,128])}}, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Flatten())
    model.add(Dense(64))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(5))
    model.add(Activation({{choice(['relu', 'sigmoid'])}}))
    model.compile(loss={{choice(['categorical_crossentropy', 'binary_crossentropy'])}},
                  optimizer='adam',
                  metrics=['accuracy'])

    model.fit_generator(
        train_generator,
        steps_per_epoch=nb_train_samples // batch_size,
        epochs=epochs,
        validation_data=validation_generator,
        validation_steps=nb_validation_samples // batch_size)

    #score, acc = model.evaluate(validation_generator)# this is for np.array inputs

    #using fit_generator
    score, acc = model.evaluate_generator(generator=validation_generator, 
                                      steps=nb_validation_samples // batch_size)

    return {'loss': -acc, 'status': STATUS_OK, 'model': model}

def data():
    img_width, img_height = 250, 1500
    batch_size = 32
    train_data_dir = 'column_data/train'
    validation_data_dir = 'column_data/validation'

    test_datagen = ImageDataGenerator(
        rescale=None)

    train_datagen = ImageDataGenerator(
        rescale=None,
        shear_range=0.2,
        horizontal_flip=False)

    train_generator = train_datagen.flow_from_directory(
        train_data_dir,
        target_size=(img_width, img_height),
        batch_size=batch_size,
        class_mode='categorical')

    validation_generator = test_datagen.flow_from_directory(
        validation_data_dir,
        target_size=(img_width, img_height),
        batch_size=batch_size,
        class_mode='categorical')

    return train_generator, validation_generator

if __name__ == '__main__':

    train_generator, validation_generator = data()

    best_run, best_model = optim.minimize(model=model,
                                          data=data,
                                          algo=tpe.suggest,
                                          max_evals=5,
                                          trials=Trials())

    print("Evaluation of best performing model:")

    print(best_model.evaluate(validation_generator))

