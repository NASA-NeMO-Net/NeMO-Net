import keras
import keras.backend as K
import numpy as np

from hyperopt import fmin, tpe, hp, STATUS_OK, Trials


def mean_categorical_crossentropy(y_true, y_pred):
    if K.image_data_format() == 'channels_last':
        loss = K.mean(keras.losses.categorical_crossentropy(y_true, y_pred), axis=[1, 2])
    elif K.image_data_format() == 'channels_first':
        loss = K.mean(keras.losses.categorical_crossentropy(y_true, y_pred), axis=[2, 3])
    return loss

def flatten_categorical_crossentropy(classes):
    def f(y_true, y_pred):
        y_true = K.reshape(y_true, (-1, classes))
        y_pred = K.reshape(y_pred, (-1, classes))
        return keras.losses.categorical_crossentropy(y_true, y_pred)
    return f

def charbonnierLoss(y_true, y_pred):
    return K.mean(K.sqrt(K.square(y_pred - y_true) + K.constant(1e-6)),-1)

def unsupervised_distance_loss(y_fixed, gamma=1):
    # Pass in y_fixed, which is a set of n_classes x n_channels fixed points. Note that n_classes can be increased beyond the original # of classes if need be
    def loss(y_true, y_pred):
        # There are original B inputs, of form B x H x R x C
        # y_true will be N x C original values (same as batch_x from NeMO_generator)
        # y_pred will be N x C values predicted by the CNN

        # Let's see if this will work, or if y_true and y_pred HAVE to be exactly identical
        M = y_fixed.shape[0]
        n_channels = y_fixed.shape[1]
        N = y_pred.shape[0] # if y_pred is a tensor, N will be ?

        # Ky_fixed = K.repeat_elements(K.expand_dims(K.variable(y_fixed),axis=-1),N,-1)
        Ky_pred = K.repeat_elements(K.reshape(K.transpose(y_pred),(1,n_channels,-1)),M,0)
        Ky_fixed = K.expand_dims(K.variable(y_fixed),-1)

        dist = K.sqrt(K.sum(K.square(Ky_fixed-Ky_pred), axis=1, keepdims=True)) 
        sum_exp_dist_yf = K.sum(K.exp(-dist), axis=0, keepdims=True) # sum of exponential of distance across yt
        p_pred = 1/K.repeat_elements(sum_exp_dist_yf,M,0)*K.exp(-dist)
        p_pred = K.reshape(p_pred,(M,-1))
        dot_p_pred = K.dot(K.transpose(p_pred),p_pred)


        ky_true = K.reshape(K.transpose(y_true),(1,n_channels,-1))
        orig_dist = K.pow(K.sqrt(K.sum(K.square(K.expand_dims(y_true,-1)-ky_true), axis=1, keepdims=False)),gamma)
        toreturn = K.sum(dot_p_pred*orig_dist)
        return toreturn
    return loss
