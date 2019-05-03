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

from NeMO_layers import CroppingLike2D, BilinearUpSampling2D, GradientReversal
from keras.regularizers import l2
from keras.layers.convolutional import Conv2D, AveragePooling2D
from NeMO_encoders import VGG16, VGG19, Alex_Encoder, Recursive_Hyperopt_Encoder
from NeMO_decoders import VGGDecoder, VGGUpsampler, VGG_DecoderBlock
from NeMO_backend import get_model_memory_usage
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials
from keras.layers import Lambda

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
    # Note that classes is never used for Recursive_Hyperopt_Encoder!!! Manually enter it in at layer level!
    encoder = Recursive_Hyperopt_Encoder(inputs, classes=classes, weight_decay=weight_decay, weights=weights, trainable=trainable_encoder, 
        conv_layers=conv_layers, full_layers=full_layers, conv_params=conv_params)
    encoder_output = encoder.outputs[0]

    return Model(inputs=inputs, output=encoder_output)

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

def SRModel_FeatureWise(hr_input_shape, lr_input_shape, SRModel, FeatureModel, FeatureLayerName):
    hr_inputs = Input(shape=hr_input_shape)
    lr_inputs = Input(shape=lr_input_shape)
    
    SR_output = SRModel(lr_inputs)
    
    Feature_layer = FeatureModel.get_layer(FeatureLayerName).output
    newFeatureModel = Model(FeatureModel.input, outputs=Feature_layer)
    keras.utils.layer_utils.print_summary(newFeatureModel, line_length=150, positions=[.35, .55, .65, 1.])
    keras.utils.layer_utils.print_summary(SRModel, line_length=150, positions=[.35, .55, .65, 1.])
    # Set FeatureModel as untrainable
    for l in newFeatureModel.layers:
        l.trainable=False
    
#     FeatureModel_content = Model(inputs=hr_inputs, outputs=Feature_layer)
#     keras.utils.layer_utils.print_summary(FeatureModel_content, line_length=150, positions=[.35, .55, .65, 1.])
    
    FeatureModel_hr_content = newFeatureModel(hr_inputs)
    FeatureModel_SR_content = newFeatureModel(SR_output)
    
#     loss = K.sqrt(K.mean((FeatureModel_hr_content - FeatureModel_SR_content)**2, (1,2))) 
    loss = Lambda(lambda x: K.sqrt(K.mean((x[0]-x[1])**2, (1,2))))([FeatureModel_hr_content, FeatureModel_SR_content])
    
    return Model([lr_inputs, hr_inputs], loss)

def DANN_Model(source_input_shape, source_model, domain_model, FeatureLayerName):
    source_inputs = Input(shape=source_input_shape)
    
    feature_layer = source_model.get_layer(FeatureLayerName).output
    index = None
    for idx, layer in enumerate(source_model.layers):
        if layer.name == FeatureLayerName:
            index = idx
            break
    
    classifier_input = Input(source_model.layers[index+1].input_shape[1:])
    classifier_layer = classifier_input
    layer_dict = {source_model.layers[index].name: classifier_input}
    for layer in source_model.layers[index+1:]:
        print("layer: ", layer.name)
        predecessor_layers = layer.input
        if type(predecessor_layers) is list: # assume maximum of 2 input layers at most
            print(layer.input[0].name)
            print(layer.input[1].name)
            classifier_layer = layer([layer_dict[layer.input[0].name.split('/')[0]],layer_dict[layer.input[1].name.split('/')[0]]])
            layer_dict.update({layer.name:classifier_layer})
        else:
            predecessor_name = layer.input.name.split('/')[0]
            classifier_layer = layer(layer_dict[predecessor_name])
            layer_dict.update({layer.name:classifier_layer})
        
    classifier_model = Model(inputs=classifier_input, outputs=classifier_layer)
#     classifier_input = Input(model.layers[index+1].input_shape[1:])
#     classifer_model = Model(input=source_model.layers[index+1].input, output=source_model.layers[-1].output)

    feature_extractor = Model(source_model.input, feature_layer)
    feature_output = feature_extractor(source_inputs)
    # Flip = GradientReversal(1.0)
    # temp_output = Flip(temp_output)
    
    domain_output = domain_model(feature_output)
    classifier_output = classifier_model(feature_output)
    
    return Model(inputs=source_inputs, outputs=[classifier_output,domain_output])


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
