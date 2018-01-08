"""Fully Convolutional Neural Networks."""
from __future__ import (
    absolute_import,
    unicode_literals
)
import keras
import tensorflow as tf
import keras.backend as K
from keras.models import Model
from keras.layers import Input, Flatten, Activation, Reshape, Dense

from NeMO_encoders import VGG16, VGG19, Alex_Encoder, Res34_Encoder
from NeMO_decoders import VGGDecoder, VGGUpsampler

<<<<<<< HEAD
def AlexNet(input_shape, classes, weight_decay=0., trainable_encoder=True, weights=None):
    inputs = Input(shape=input_shape)

    encoder = Alex_Encoder(inputs, classes=classes, weight_decay=weight_decay, weights=weights, trainable=trainable_encoder)
    encoder_output = encoder.outputs[0]
    scores = Dense(classes, activation = 'softmax')(encoder_output)

    return Model(inputs=inputs, outputs=scores)

def ResNet34(input_shape, classes, weight_decay=0., trainable_encoder=True, weights=None):
    """ Normal Resnet34 
    """
    inputs = Input(shape=input_shape)

    encoder = Res34_Encoder(inputs, classes=classes, weight_decay=weight_decay, weights=weights, trainable=trainable_encoder, fcflag=True)
    encoder_output = encoder.outputs[0] # Only take last output
    scores = Activation('softmax')(encoder_output)

    return Model(inputs=inputs, outputs=scores)
# scores = Activation('softmax')(outputs)
=======
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials

>>>>>>> Rm122_KerasCNN

def FCN(*args, **kwargs):
    """Fully Convolutional Networks for semantic segmentation with VGG16.

    # Arguments
        input_shape: input image shape
        classes: number of classes
        trainable_encoder: Bool whether the weights of encoder are trainable
        weights: pre-trained weights to load (None if training from scratch)
    # Returns
        A Keras model instance

    """
    return FCN_VGG16(*args, **kwargs)


def FCN_VGG16(input_shape, classes, weight_decay=0.,
              trainable_encoder=True, weights=None):
    """Fully Convolutional Networks for semantic segmentation with VGG16.

    # Arguments
        input_shape: input image shape
        classes: number of classes
        trainable_encoder: Bool whether the weights of encoder are trainable
        weights: pre-trained weights to load (None for training from scratch)



    # Returns
        A Keras model instance

    """
    # input
    inputs = Input(shape=input_shape)

    # Get the feature pyramid [drop7, pool4, pool3] from the VGG16 encoder
    pyramid_layers = 3
    encoder = VGG16(inputs, weight_decay=weight_decay,
                    weights=weights, trainable=trainable_encoder)
    feat_pyramid = encoder.outputs[:pyramid_layers]

    # Append image to the end of feature pyramid
    feat_pyramid.append(inputs)

    # Decode feature pyramid
    outputs = VGGUpsampler(feat_pyramid, scales=[1, 1e-2, 1e-4], classes=classes, weight_decay=weight_decay)

    # Activation TODO{jihong} work only for channels_last
    scores = Activation('softmax')(outputs)

    # return model
    return Model(inputs=inputs, outputs=scores)


def FCN_VGG19(input_shape, classes, weight_decay=0,
              trainable_encoder=True, weights='imagenet'):
    """Fully Convolutional Networks for semantic segmentation with VGG16.

    # Arguments
        input_shape: input image shape
        classes: number of classes
        trainable_encoder: Bool whether the weights of encoder are trainable
        weights: pre-trained weights to load (None for training from scratch)



    # Returns
        A Keras model instance

    """
    # input
    inputs = Input(shape=input_shape)

    # Get the feature pyramid [drop7, pool4, pool3] from the VGG16 encoder
    pyramid_layers = 3
    encoder = VGG19(inputs, weight_decay=weight_decay,
                    weights='imagenet', trainable=trainable_encoder)
    feat_pyramid = encoder.outputs[:pyramid_layers]

    # Append image to the end of feature pyramid
    feat_pyramid.append(inputs)

    # Decode feature pyramid
    outputs = VGGUpsampler(feat_pyramid, scales=[1, 1e-2, 1e-4], classes=classes, weight_decay=weight_decay)

    # Activation TODO{jihong} work only for channels_last
    outputs = Activation('softmax')(outputs)

    # return model
    return Model(inputs=inputs, outputs=outputs)
