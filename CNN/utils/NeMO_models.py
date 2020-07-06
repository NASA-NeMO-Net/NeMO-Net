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

from NeMO_layers import CroppingLike2D, BilinearUpSampling2D, GradientReversal, Batch_Split, Gram_Loss, Var_Loss, Content_Loss
from keras.regularizers import l2
from keras.layers.convolutional import Conv2D, AveragePooling2D
from NeMO_encoders import VGG16, VGG19, Alex_Encoder, Recursive_Hyperopt_Encoder
from NeMO_decoders import VGGDecoder, VGGUpsampler, VGG_DecoderBlock
from NeMO_backend import get_model_memory_usage
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials
from keras.layers import Lambda, Dense, Flatten, LeakyReLU

def content_loss(base, combination):
    return K.sum(K.square(combination - base))

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


def AlexNetLike(input_shape, classes, weight_decay=0., trainable_encoder=True, weights=None, conv_layers=0, full_layers=0, conv_params=None, onebyoneconv= True, reshape=True):
    inputs = Input(shape=input_shape)
    # Note that classes is never used for Recursive_Hyperopt_Encoder!!! Manually enter it in at layer level!
    encoder = Recursive_Hyperopt_Encoder(inputs, classes=classes, weight_decay=weight_decay, weights=weights, trainable=trainable_encoder, 
        conv_layers=conv_layers, full_layers=full_layers, conv_params=conv_params)
    encoder_output = encoder.outputs[0]
    if onebyoneconv:
        final_1b1conv = Conv2D(classes, (1,1), padding="same", kernel_initializer='he_normal', kernel_regularizer=l2(weight_decay), name='final_1b1conv')(encoder_output)
    else:
        final_1b1conv = encoder_output
        
    if reshape:
        scores = Activation('softmax')(final_1b1conv)
        scores = Reshape((input_shape[0]*input_shape[1], classes))(scores)  # for class weight purposes, (sample_weight_mode: 'temporal')
    else:
        scores = final_1b1conv

    return Model(inputs=inputs, output=scores)

def Discriminator_AlexNetLike(input_shape, classes, weight_decay=0., trainable_encoder=True, weights=None, conv_layers=0, full_layers=0, conv_params=None):
    inputs = Input(shape=input_shape)
    # Note that classes is never used for Recursive_Hyperopt_Encoder!!! Manually enter it in at layer level!
    encoder = Recursive_Hyperopt_Encoder(inputs, classes=classes, weight_decay=weight_decay, weights=weights, trainable=trainable_encoder, 
        conv_layers=conv_layers, full_layers=full_layers, conv_params=conv_params)
    encoder_output = encoder.outputs[0]
    
    x = Flatten()(encoder_output)

    x = Dense(1024)(x)
    x = LeakyReLU(alpha=0.2)(x)
    x = Dense(1, activation='sigmoid')(x)
    
    return Model(inputs=inputs, output=x, name='model_discriminator')
    

def SharpMask_FCN(input_shape, classes, decoder_index, weight_decay=0., trainable_encoder=True, weights=None, conv_layers=5, full_layers=0, conv_params=None, 
    scales = 1, bridge_params=None, prev_params=None, next_params=None, reshape=True):
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

    if reshape:
        scores = Activation('softmax')(final_1b1conv)
        scores = Reshape((input_shape[0]*input_shape[1], classes))(scores)  # for class weight purposes, (sample_weight_mode: 'temporal')
    else:
        scores = Activation('softmax')(final_1b1conv)

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


def StyleTransfer(feature_input_shape, product_input_shape, TransferModel, FeatureModel, FeatureLayers, ContentLayers, style_weight, variation_weight):
    feature_input = Input(shape=feature_input_shape) # source
    product_input = Input(shape=product_input_shape) # target
    
    product_output = TransferModel(product_input)
    
    content_weight = 1 - style_weight
    
    featurelayer_output = []
    for layer in FeatureLayers:
        featurelayer_output.append(FeatureModel.get_layer(layer).output)
    newFeatureModel = Model(FeatureModel.input, outputs=featurelayer_output)
    # Set FeatureModel as untrainable
    for l in newFeatureModel.layers:
        l.trainable=False
    
    # Not as efficient, but clean since it keeps style and content losses in separate models
    contentlayer_output = []
    for layer in ContentLayers:
        contentlayer_output.append(FeatureModel.get_layer(layer).output)
    newContentModel = Model(FeatureModel.input, outputs=contentlayer_output)
    # Set ContentModel as untrainable
    for l in newContentModel.layers:
        l.trainable=False
        
    newFeatureModel_feature_content = newFeatureModel(feature_input)
    newFeatureModel_prod_content = newFeatureModel(product_output) # expected output: list of tensors of shape (?,y,x,channels)
    newContentModel_feature_content = newContentModel(product_input) # take original target and put it through the content model 
    newContentModel_prod_content = newContentModel(product_output)
    
    loss = []
#     loss.append(Var_Loss(variation_weight)(product_output))
    # Style Loss (in the form of Gram Loss)
    if type(newFeatureModel_feature_content) is list: # list of output features
        for i in range(len(newFeatureModel_feature_content)): # go through every output feature in list
            templist = [newFeatureModel_feature_content[i], newFeatureModel_prod_content[i]]
            loss.append(Gram_Loss(style_weight/len(FeatureLayers))(templist))
    
    # Content Loss
    loss.append(Content_Loss(content_weight)([newContentModel_feature_content, newContentModel_prod_content]))
            
    added_loss = keras.layers.Add()(loss)
    return Model([feature_input, product_input], added_loss)

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

def SRGAN(input_shape, generator, discriminator):
    discriminator.trainable = False

    x_in = Input(shape=input_shape)
    x_gen = generator(x_in)
    x_dis = discriminator(x_gen)

    return Model(x_in, [x_gen, x_dis])

def DANN_Model(source_input_shape, source_model, domain_model, FeatureLayerName):
    source_inputs = Input(shape=source_input_shape)
    index = 0
    
    layer_dict = {}
#     classifier_input = Input(source_model.layers[index+1].input_shape[1:])
#     layer_dict = {source_model.layers[0].name: source_model.layers[0].input}
#     print(source_model.layers[index+1].input_shape[1:])
    
    for layer in source_model.layers[index+1:]:
        predecessor_layers = layer.input
                      
        if type(predecessor_layers) is list: # assume maximum of 2 input layers at most
            classifier_layer = layer([layer_dict[layer.input[0].name.split('/')[0]],layer_dict[layer.input[1].name.split('/')[0]]])
            layer_dict.update({layer.name:classifier_layer})
        else:
            predecessor_name = layer.input.name.split('/')[0] # if another layer with the same input name exists, a '_#' is appended automatically
            if "input" in predecessor_name:
                classifier_layer = layer(source_inputs)
            else:
#                 if predecessor_name == FeatureLayerName: # if end of feature layer, split and take first half
#                     TestSplit = Batch_Split()
#                     split_layer = TestSplit(layer_dict[predecessor_name])
#                     classifier_layer = layer(split_layer)
#                 else:
                classifier_layer = layer(layer_dict[predecessor_name])
            layer_dict.update({layer.name:classifier_layer})
                
    d_index = 0
    for dlayer in domain_model.layers[d_index+1:]:
        dpredecessor_layers = dlayer.input
        if dpredecessor_layers is list:
            print("")
        else:
            dpredecessor_name = dlayer.input.name.split('/')[0]
            if "input" in dpredecessor_name:
                dpredecessor_name = FeatureLayerName
                Flip = GradientReversal(1.0)

                dclassifier_layer = Flip(layer_dict[dpredecessor_name])
                layer_dict.update({dclassifier_layer.name:dclassifier_layer})
                dpredecessor_name = dclassifier_layer.name
                
            dclassifier_layer = dlayer(layer_dict[dpredecessor_name])
            count = 0
            templayername = dlayer.name
            while templayername in layer_dict:
                count = count + 1
                templayername = dlayer.name + '_' + str(count)
            dlayer.name = templayername 

            if count > 0:
                layer_dict.update({templayername:dclassifier_layer})
            else:
                layer_dict.update({dlayer.name:dclassifier_layer})

    
    classifier_output = classifier_layer
    domain_output = dclassifier_layer
    
    return Model(inputs=source_inputs, outputs=[classifier_output, domain_output])


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
