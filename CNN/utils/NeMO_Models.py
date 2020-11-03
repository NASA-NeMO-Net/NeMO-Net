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

from NeMO_layers import CroppingLike2D, BilinearUpSampling2D, GradientReversal, Batch_Split, Gram_Loss, Var_Loss, Content_Loss
from NeMO_Encoders import Recursive_Encoder
from NeMO_Decoders import NeMO_Decoder
from NeMO_backend import get_model_memory_usage
  

def NeMO_FCN(input_shape: Tuple[int, int, int], 
	classes: int, 
	decoder_index: List[int], 
	weight_decay: float = 0., 
	trainable_encoder: bool = True, 
	conv_blocks: int = 5, 
	dense_blocks: int = 0,
	conv_params: Dict[str, List] = None, 
	scales: Union[List[float], float] = 1.0,
	bridge_params: Dict[str, List] = None, 
	prev_params: Dict[str, List] = None, 
	next_params: Dict[str, List] = None, 
	reshape: bool = True):
	''' Generic structure for Fully Convolutional Network, with one final layer at the end for class prediction
	
		input_shape: 3D tensor if input shape
		classes: # of classes
		decoder_index: Which encoder outputs to take when passing it into the decoder
		weight_decay: Weight decay
		trainable_encoder: Whether encoder is trainable
		conv_blocks: # of convolutional blocks
		dense_blocks: # of dense blocks
		conv_params: Convolutional parameters
		scales: How much to scale bridge block output before combined with prev block in the decoder
		bridge_params: Bridge block parameters
		prev_params: Prev block parameters
		Next_params: Next block parameters
		reshape: Whether to reshape for class weight purposes
	'''

	inputs = Input(shape=input_shape)
	pyramid_layers = decoder_index

	encoder = Recursive_Encoder(inputs, 
		classes=classes, 
		weight_decay=weight_decay, 
		trainable=trainable_encoder, 
		conv_blocks=conv_blocks,
		dense_blocks=dense_blocks, 
		conv_params=conv_params)


	feat_pyramid = [encoder.outputs[index] for index in pyramid_layers]
	# Append image to the end of feature pyramid
	feat_pyramid.append(inputs)

	# Decode feature pyramid
	outputs = NeMO_Decoder(feat_pyramid,  
		classes=classes, 
		scales=scales, 
		weight_decay=weight_decay, 
		bridge_params=bridge_params,
		prev_params=prev_params, 
		next_params=next_params)

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
