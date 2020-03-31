from __future__ import (
	absolute_import,
	unicode_literals
)
import numpy as np
import copy
import keras
import keras.backend as K

#from keras.models import Model
from keras.engine.training import Model
from keras.utils.data_utils import get_file
from keras.utils import layer_utils

from keras.layers import Cropping2D, Concatenate, Add
from NeMO_blocks import (
	alex_conv,
	alex_fc,
	parallel_conv,
	vgg_convblock,
	vgg_fcblock,
	pool_concat,
	res_initialconv,
	res_basicconv,
	res_megaconv,
	res_1b1conv,
	res_fc,
	vgg_conv,
	vgg_fc
)
from NeMO_backend import load_weights

from hyperopt import fmin, tpe, hp, STATUS_OK, Trials

def Func_Encoder(inputs, blocks):

	if type(inputs) is list:
		inputs_copy = [np.copy(inp) for inp in inputs]
	else:
		inputs_copy = inputs

	for i, block in enumerate(blocks):
		if i == 0:
			x = block(inputs_copy)
		else:
			x = block(x)
	return x

def Func_Parallel_Hyperopt_Encoder(inputs, classes, parallel_layers=4, combine_method='concat', conv_params=None, weight_decay=0.):

# double brackets to signify all parallel filters follow same parameters
	default_conv_params = {"filters": [[1024,1024,classes]],
		"conv_size": [[(3,3),(1,1),(1,1)]],
		"conv_strides":  [[(3,3),(1,1),(1,1)]],
		"padding": ['same','same','same','same'],
		"dilation_rate": [[(6,6),(1,1),(1,1)], [(12,12),(1,1),(1,1)], [(18,18),(1,1),(1,1)], [(24,24),(1,1),(1,1)]],
		"pool_size": [[(2,2),(2,2),(2,2)]],
		"pool_strides": [[(2,2),(2,2),(2,2)]],
		"pad_size": [(6,6), (12,12), (18,18), (24,24)],
		"layercombo": ["zcadcadc","zcadcadc","zcadcadc","zcadcadc"],
		"full_filters": [4096,2048],
		"dropout": [0.5,0.5]}
	filters = conv_params["filters"]
	conv_size = conv_params["conv_size"]
	conv_strides = conv_params["conv_strides"]
	padding = conv_params["padding"]
	dilation_rate = conv_params["dilation_rate"]
	pool_size = conv_params["pool_size"]
	pool_strides = conv_params["pool_strides"]
	pad_size = conv_params["pad_size"]
	layercombo = conv_params["layercombo"]

	# f = lambda input,c_count: input[0] if len(input)==1 else input[c_count]
	# actual start of CNN
	blocks = []
	block_name = 'parallel_block'
	block = parallel_conv(filters, conv_size, conv_strides=conv_strides, padding=padding, pad_bool=True, pad_size=pad_size, 
	pool_size=pool_size, pool_strides=pool_strides, dilation_rate=dilation_rate, dropout=[0.5], layercombo=layercombo,
	weight_decay=weight_decay, block_name=block_name)
	blocks.append(block)

	if combine_method == "concat":
		block = Concatenate(axis=-1)
		blocks.append(block)
	elif combine_method == "add":
		block = Add()
		blocks.append(block)

	return Func_Encoder(inputs=inputs, blocks=blocks)