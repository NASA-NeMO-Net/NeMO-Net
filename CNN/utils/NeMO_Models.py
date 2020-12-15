"""Fully Convolutional Neural Networks."""
from __future__ import (
    absolute_import,
    unicode_literals
)

from typing import Tuple, Callable, List, Union, Dict

import numpy as np
import keras
import tensorflow as tf
import keras.backend as K
from keras.models import Model
from keras.layers import Input, Flatten, Activation, Reshape, Dense, Cropping2D, Lambda, LeakyReLU
from keras.regularizers import l2
from keras.layers.convolutional import Conv2D, AveragePooling2D

from NeMO_Layers import CroppingLike2D, BilinearUpSampling2D
from NeMO_Encoders import Recursive_Encoder
from NeMO_Decoders import NeMO_Decoder
from NeMO_Architecture import NeMO_FCN_Architecture, NeMO_Architecture
from NeMO_Backend import get_model_memory_usage
  

# Need to input architecture in here!
def NeMO_FCN(input_shape: Tuple[int, int, int], 
	classes: int, 
	architecture: NeMO_FCN_Architecture,
	weight_decay: float = 0., 
	trainable_encoder: bool = True, 
	reshape: bool = True):
	''' Generic structure for Fully Convolutional Network, with one final layer at the end for class prediction
	
		input_shape: 3D tensor if input shape
		classes: # of classes
		architecture: Specific architecture of NeMO_FCN. Look at NeMO_Architecture to define.
		weight_decay: Weight decay
		trainable_encoder: Whether encoder is trainable
		reshape: Whether to reshape for class weight purposes
	'''

	inputs = Input(shape=input_shape)
	pyramid_layers = architecture.decoder_index

	encoder = Recursive_Encoder(inputs, 
		classes = classes, 
		weight_decay = weight_decay, 
		trainable = trainable_encoder, 
		conv_blocks = architecture.conv_blocks,
		dense_blocks = architecture.dense_blocks, 
		conv_params = architecture.conv_params)


	feat_pyramid = [encoder.outputs[index] for index in pyramid_layers]
	# Append image to the end of feature pyramid
	feat_pyramid.append(inputs)

	# Decode feature pyramid
	outputs = NeMO_Decoder(feat_pyramid,  
		classes = classes, 
		scales = architecture.scales, 
		weight_decay = weight_decay, 
		bridge_params = architecture.bridge_params,
		prev_params = architecture.prev_params, 
		next_params = architecture.next_params)

	final_1b1conv = Conv2D(classes, 
		(1,1), 
		padding="same", 
		kernel_initializer='he_normal', 
		kernel_regularizer=l2(weight_decay), 
		name='final_1b1conv')(outputs)

	if reshape:
		scores = Activation('softmax')(final_1b1conv)
		scores = Reshape((input_shape[0]*input_shape[1], classes))(scores)  # for class weight purposes, (sample_weight_mode: 'temporal')
	else:
		scores = Activation('softmax')(final_1b1conv)

	return Model(inputs=inputs, outputs=scores)
