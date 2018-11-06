"""Fully Convolutional Neural Networks."""
from __future__ import (
    absolute_import,
    unicode_literals
)
import numpy as np
import keras
import tensorflow as tf
import keras.backend as K
from keras.models import Model
from keras.layers import Input, Flatten, Activation, Reshape, Dense, Cropping2D

from NeMO_layers import CroppingLike2D, BilinearUpSampling2D
from keras.regularizers import l2
from keras.layers.convolutional import Conv2D, AveragePooling2D
from NeMO_encoders import VGG16, VGG19, Alex_Encoder, Res34_Encoder, Alex_Hyperopt_Encoder, Alex_Parallel_Hyperopt_Encoder, VGG_Hyperopt_Encoder, Recursive_Hyperopt_Encoder
from NeMO_decoders import VGGDecoder, VGGUpsampler, VGG_DecoderBlock
from NeMO_functional_encoders import Func_Parallel_Hyperopt_Encoder
from NeMO_backend import get_model_memory_usage
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials

def AlexNet(input_shape, classes, weight_decay=0., trainable_encoder=True, weights=None):
    inputs = Input(shape=input_shape)

    encoder = Alex_Encoder(inputs, classes=classes, weight_decay=weight_decay, weights=weights, trainable=trainable_encoder)
    encoder_output = encoder.outputs[0]
    scores = Dense(classes, activation = 'softmax')(encoder_output)

    return Model(inputs=inputs, outputs=scores)

# def Alex_Hyperopt_ParallelNet(input_shape, crop_shapes, classes, weight_decay=0., trainable_encoder=True, weights=None, conv_layers=3, full_layers=1, conv_params=None):
#     inputs=Input(shape=input_shape)

#     encoder = Alex_Parallel_Hyperopt_Encoder(inputs, crop_shapes=crop_shapes, classes=classes, weight_decay=weight_decay, weights=weights, trainable=trainable_encoder,
#         conv_layers=conv_layers, full_layers=full_layers, conv_params=conv_params)
#     encoder_output = encoder.outputs[0]
#     scores = Dense(classes, activation = 'softmax')(encoder_output)

#     return Model(inputs=inputs, outputs=scores)

def TestModel(input_shape, classes, decoder_index, weight_decay=0., trainable_encoder=True, weights=None, conv_layers=1, full_layers=0, conv_params=None,
    scales=1, bridge_params=None, prev_params=None, next_params=None):

    inputs = Input(shape=input_shape)
    pyramid_layers = decoder_index

    encoder = Recursive_Hyperopt_Encoder(inputs, classes=classes, weight_decay=weight_decay, weights=weights, trainable=trainable_encoder, conv_layers=conv_layers,
        full_layers=full_layers, conv_params=conv_params)
    # outputs1 = Recursive_Hyperopt_Encoder(inputs)?
    # outputs2 = Recursive_Hyperopt_Encoder(outputs1[0])?

    feat_pyramid = [encoder.outputs[index] for index in pyramid_layers]
    feat_pyramid.insert(0,inputs)

    # Decode feature pyramid
    outputs = VGG_DecoderBlock(feat_pyramid,  classes=classes, scales=scales, weight_decay=weight_decay, 
        bridge_params=bridge_params, prev_params=prev_params, next_params=next_params)
 
    # final_1b1conv = Conv2D(classes, (1,1), padding="same", kernel_initializer='he_normal', kernel_regularizer=l2(weight_decay), name='final_1b1conv')(outputs)

    # scores = Activation('softmax')(final_1b1conv)
    # scores = Reshape((input_shape[0]*input_shape[1], classes))(scores)  # for class weight purposes

    return Model(inputs=inputs, outputs=outputs)


def AlexNetLike(input_shape, classes, weight_decay=0., trainable_encoder=True, weights=None, conv_layers=0, full_layers=0, conv_params=None):
    inputs = Input(shape=input_shape)
    encoder = Recursive_Hyperopt_Encoder(inputs, classes=classes, weight_decay=weight_decay, weights=weights, trainable=trainable_encoder, 
        conv_layers=conv_layers, full_layers=full_layers, conv_params=conv_params)
    encoder_output = encoder.outputs[0]

    return Model(inputs=inputs, output=encoder_output)

def VGG16_DeepLabV2(input_shape, classes, weight_decay=0., trainable_encoder=True, batch_size=1, weights=None, conv_layers=5, full_layers=0, conv_params=None, 
    parallel_layers=4, parallelconv_params=None):
    inputs1 = Input(shape=input_shape)
    encoder1 = VGG_Hyperopt_Encoder(inputs1, classes=classes, weight_decay=weight_decay, weights=weights, trainable=trainable_encoder, conv_layers=conv_layers,
        full_layers=full_layers, conv_params=conv_params)
    print("Memory required (GB): ", get_model_memory_usage(batch_size, encoder1))

    tempoutput = encoder1.outputs[0]
    # output1shape = tuple([int(tempoutput.shape[i]) for i in range(1,len(tempoutput.shape))])

    # inputs2 = Input(shape=output1shape)
    # encoder2 = Alex_Parallel_Hyperopt_Encoder(inputs2, classes=classes, parallel_layers=parallel_layers, combine_method='add', 
    #     conv_params=parallelconv_params, weight_decay = weight_decay, weights=weights, trainable = trainable_encoder)
    # print("Memory required (GB): ", get_model_memory_usage(batch_size, encoder2))

    outputs = Func_Parallel_Hyperopt_Encoder(tempoutput, classes=classes, parallel_layers=parallel_layers, combine_method='add', 
        conv_params=parallelconv_params, weight_decay = weight_decay)

    upscore = BilinearUpSampling2D(target_shape=K.int_shape(inputs1), name='BilinearUpsample')(outputs)

    scores = Activation('softmax')(upscore)
    scores = Reshape((input_shape[0]*input_shape[1], classes))(scores) # for class weight purposes
    # print(encoder2(encoder1.output[0])[0])

    return Model(inputs=encoder1.input, outputs=scores)


def VGG_Hyperopt_FCN(input_shape, classes, decoder_index, weight_decay=0., trainable_encoder=True, weights=None, conv_layers=5, full_layers=1, conv_params=None, deconv_params=None):
    inputs = Input(shape=input_shape)
    pyramid_layers = decoder_index

    encoder = VGG_Hyperopt_Encoder(inputs, classes=classes, weight_decay=weight_decay, weights=weights, trainable=trainable_encoder, conv_layers=conv_layers,
        full_layers=full_layers, conv_params=conv_params)


    feat_pyramid = [encoder.outputs[index] for index in pyramid_layers]
    # Append image to the end of feature pyramid
    feat_pyramid.append(inputs)

    # Decode feature pyramid
    outputs = VGG_DecoderBlock(feat_pyramid,  classes=classes, weight_decay=weight_decay, deconv_params=deconv_params)

    scores = Activation('softmax')(outputs)
    scores = Reshape((input_shape[0]*input_shape[1], classes))(scores)  # for class weight purposes

    # return model
    return Model(inputs=inputs, outputs=scores)

def SharpMask_FCN(input_shape, classes, decoder_index, weight_decay=0., trainable_encoder=True, weights=None, conv_layers=5, full_layers=0, conv_params=None, 
    scales = 1, bridge_params=None, prev_params=None, next_params=None):
    inputs = Input(shape=input_shape)
    pyramid_layers = decoder_index

    encoder = Recursive_Hyperopt_Encoder(inputs, classes=classes, weight_decay=weight_decay, weights=weights, trainable=trainable_encoder, conv_layers=conv_layers,
        full_layers=full_layers, conv_params=conv_params)


    feat_pyramid = [encoder.outputs[index] for index in pyramid_layers]
    # Append image to the end of feature pyramid
    feat_pyramid.append(inputs)

    # Decode feature pyramid
    outputs = VGG_DecoderBlock(feat_pyramid,  classes=classes, scales=scales, weight_decay=weight_decay, 
        bridge_params=bridge_params, prev_params=prev_params, next_params=next_params)

    final_1b1conv = Conv2D(classes, (1,1), padding="same", kernel_initializer='he_normal', kernel_regularizer=l2(weight_decay), name='final_1b1conv')(outputs)

    scores = Activation('softmax')(final_1b1conv)
    scores = Reshape((input_shape[0]*input_shape[1], classes))(scores)  # for class weight purposes, (sample_weight_mode: 'temporal')

    # return model
    return Model(inputs=inputs, outputs=scores)

def TestModel_EncoderDecoder(input_shape, classes, decoder_index, weight_decay=0., trainable_encoder=True, weights=None, conv_layers=5, full_layers=0, conv_params=None, 
    scales = 1, bridge_params=None, prev_params=None, next_params=None):
    inputs = Input(shape=input_shape)
    pyramid_layers = decoder_index

    encoder = Recursive_Hyperopt_Encoder(inputs, classes=classes, weight_decay=weight_decay, weights=weights, trainable=trainable_encoder, conv_layers=conv_layers,
        full_layers=full_layers, conv_params=conv_params)

    feat_pyramid = [encoder.outputs[index] for index in pyramid_layers]
    # Append image to the end of feature pyramid
    feat_pyramid.append(inputs)

    # Decode feature pyramid
    outputs = VGG_DecoderBlock(feat_pyramid,  classes=classes, scales=scales, weight_decay=weight_decay, 
        bridge_params=bridge_params, prev_params=prev_params, next_params=next_params)

    # return model
    return Model(inputs=inputs, outputs=outputs)

def ResNet34(input_shape, classes, weight_decay=0., trainable_encoder=True, weights=None):
    """ Normal Resnet34 
    """
    inputs = Input(shape=input_shape)

    encoder = Res34_Encoder(inputs, classes=classes, weight_decay=weight_decay, weights=weights, trainable=trainable_encoder, fcflag=True)
    #feat_pyramid = encoder.outputs[:pyramid_layers]
    encoder_output = encoder.outputs[0]

    scores = Activation('softmax')(encoder_output)

    return Model(inputs=inputs, outputs=scores)
# scores = Activation('softmax')(outputs)



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
