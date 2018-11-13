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
    recursive_conv,
    alex_conv,
    alex_fc,
    vgg_conv,
    vgg_fc
)
from NeMO_backend import load_weights

from hyperopt import fmin, tpe, hp, STATUS_OK, Trials


class Encoder(Model):
    """Encoder for Fully Convolutional Networks.
    :param inputs: 4D Tensor, the input tensor
    :param blocks: 1D array, list of functional convolutional blocks

    :return A Keras Model with outputs including the output of
    each block except the final conv block (using the encoder's top instead)

    >>> from keras.layers import Input
    >>> from keras_fcn.encoders import Encoder
    >>> from keras_fcn.blocks import (vgg_conv, vgg_fc)
    >>> inputs = Input(shape=(224, 224, 3))
    >>> blocks = [vgg_conv(64, 2, 'block1'),
    >>>           vgg_conv(128, 2, 'block2'),
    >>>           vgg_conv(256, 3, 'block3'),
    >>>           vgg_conv(512, 3, 'block4'),
    >>>           vgg_conv(512, 3, 'block5'),
    >>>           vgg_fc(4096)]
    >>> encoder = Encoder(inputs, blocks, weights='imagenet',
    >>>                   trainable=True)
    >>> feat_pyramid = encoder.outputs   # A feature pyramid with 5 scales

    """

    def __init__(self, inputs, blocks, weights=None,
                 trainable=True, name='encoder'):
        inverse_pyramid = []

        # convolutional block
        conv_blocks = blocks[:-1]
        for i, block in enumerate(conv_blocks):
            if i == 0:
                x = block(inputs)
                inverse_pyramid.append(x)
            elif i < len(conv_blocks) - 1:
                x = block(x)
                inverse_pyramid.append(x)
            else:
                x = block(x)

        # fully convolutional block
        fc_block = blocks[-1]
        y = fc_block(x)
        inverse_pyramid.append(y)

        outputs = list(reversed(inverse_pyramid))

        super(Encoder, self).__init__(
            inputs=inputs, outputs=outputs)

        # load pre-trained weights
        if weights is not None:
            weights_path = get_file(
                '{}_weights_tf_dim_ordering_tf_kernels.h5'.format(name),
                weights,
                cache_subdir='models')
            layer_names = load_weights(self, weights_path)
            if K.image_data_format() == 'channels_first':
                layer_utils.convert_all_kernels_in_model(self)

        # Freezing basenet weights
        if trainable is False:
            for layer in self.layers:
                if layer.name in layer_names:
                    layer.trainable = False


class Res_Encoder(Model):
    """Same as Encoder, but does not get rid of any output blocks
    :param inputs: 4D Tensor, the input tensor
    :param blocks: 1D array, list of functional convolutional blocks

    :return A Keras Model with outputs
    """

    # inputs: normal input
    # blocks: blocks that make up the encoder
    # crop_shapes: list of tuples indicating shapes to be cropped out of the center of inputs (e.g. [(5,5),(10,10)])
    # weights: weights to be loaded (if any)
    # traininable: Trainable layers
    # name: Name of encoder
    def __init__(self, inputs, blocks, weights=None, trainable=True, name='encoder'):
        inverse_pyramid = []


        # all parallel blocks
        if type(inputs) is list:
            inputs_copy = [np.copy(inp) for inp in inputs]
        else:
            inputs_copy = inputs

        for i, block in enumerate(blocks):
            if i == 0:
                x = block(inputs_copy)
            else:
                x = block(x)

            if type(x) is list:
                inverse_pyramid.append(list(x))
                # print([xi.shape for xi in x])
            else:
                inverse_pyramid.append(x)
                # print(x.shape)

        pyramid = reversed(inverse_pyramid)
        outputs = []
        for item in pyramid:
            if type(item) is list:
                for miniitem in item:
                    outputs.append(miniitem)
            else:
                outputs.append(item)
        # outputs = list(reversed(inverse_pyramid))

        super(Res_Encoder, self).__init__(inputs=inputs, outputs=outputs)
        # print(self.summary())

        # load pre-trained weights
        if weights is not None:
            weights_path = get_file(
                '{}_weights_tf_dim_ordering_tf_kernels.h5'.format(name),
                weights,
                cache_subdir='models')
            layer_names = load_weights(self, weights_path)
            if K.image_data_format() == 'channels_first':
                layer_utils.convert_all_kernels_in_model(self)

        # Freezing basenet weights
        if trainable is False:
            for layer in self.layers:
                if layer.name in layer_names:
                    layer.trainable = False

class Alex_Encoder(Res_Encoder):
    def __init__(self, inputs, classes, weight_decay=0., weights=None, trainable=True):
        # filters = [64, 128, 256, 384, 512]
        # conv_size = [(7,7),(5,5),(3,3),(3,3),(3,3)]
        # pool_size = [(2,2),(2,2),(1,1),(1,1),(2,2)]
        # pool_stride = [(2,2),(2,2),(1,1),(1,1),(2,2)]
        # pool_bool = [True, True, False, False, True]
        # pad_bool = [False, False, True, True, True]
        # batchnorm_bool = [True, True, False, False, False]
        filters = [64, 128]
        conv_size = [(7,7),(5,5)]
        pool_size = [(2,2),(2,2)]
        pool_stride = [(2,2),(2,2)]
        pool_bool = [True, True]
        pad_bool = [False, False]
        batchnorm_bool = [True, True]

        full_filters = [256]
        drop_bool = [True]
        drop_val = [0.5]
        
        blocks = []
        for i in range(len(filters)):
            block_name = 'alexblock{}'.format(i + 1)
            block = alex_conv(filters[i], conv_size[i], pad_bool=pad_bool[i], pool_bool=pool_bool[i], batchnorm_bool=batchnorm_bool[i], 
                pool_size=pool_size[i], pool_strides=pool_stride[i], weight_decay=weight_decay, block_name=block_name)
            blocks.append(block)

        for i in range(len(full_filters)):
            block_name='alexfc{}'.format(i + 1)
            if i==0:
                block = alex_fc(full_filters[i], flatten_bool=True, dropout_bool=drop_bool[i], dropout=drop_val[i], weight_decay=weight_decay, 
                    block_name=block_name)
            else:
                block = alex_fc(full_filters[i], flatten_bool=False, dropout_bool=drop_bool[i], dropout=drop_val[i], weight_decay=weight_decay, 
                    block_name=block_name)
            blocks.append(block)

        super(Alex_Encoder, self).__init__(inputs=inputs, blocks=blocks, weights=weights, trainable = trainable)

def recursive_concatcombo(layercombo):

    if type(layercombo) is str:
        return layercombo

    if type(layercombo) is tuple or list:
        a = [recursive_concatcombo(combo) for combo in layercombo]
    return a

def flatten_list(S):
    if S == []:
        return S
    if isinstance(S[0], list):
        return flatten_list(S[0]) + flatten_list(S[1:])
    return S[:1] + flatten_list(S[1:])


def load_specific_param(num_layers, conv_params, specific_param, combokey, supercombo, default_conv_params=None, layer_str="convolutional"):
    # default_param = default_conv_params[specific_param]
    # supercombo = [combo for combo in layercombo]
    param = []

    if specific_param is "layercombo":
        try:
            param = conv_params[specific_param]
        except:
            print("Please specify {} parameter of length {}".format(specific_param,num_layers))

        if len(param) == 1:     # if only one param, use that for all the layers
            param = param*num_layers

        if len(param) != num_layers:
            print("{} parameter not the same length as the # of {} layers: {}".format(specific_param,layer_str,num_layers))
            raise ValueError
    elif specific_param is "layercombine":
        try:
            param = conv_params[specific_param]
        except:
            print("{} not specified... will use 'sum' function where appropriate".format(specific_param))
            param = ['sum']*num_layers
        
        if len(param) == 1:
            param = param*num_layers
            
        if len(param) != num_layers:
            print("{} parameter not the same length as the # of {} layers: {}".format(specific_param, layer_str, num_layers))
            raise ValueError
    else:
        if combokey in supercombo:
            try:
                param = conv_params[specific_param]
            except:
                print("{} parameter not found, please specify it in the params dictionary as it is required".format(specific_param))
                raise ValueError

            if len(param) == 1:     # if only one param, use that for all the layers
                param = param*num_layers

            if len(param) != num_layers:
                print("{} parameter not the same length as the # of {} layers: {}".format(specific_param,layer_str,num_layers))
                raise ValueError
    if len(param) is 0:
        param = [None]*num_layers

    print("{}: {}".format(specific_param, param))

    return param

def load_conv_params(conv_layers, full_layers, default_conv_params, conv_params):
    print("---------------------------------------------------------")
    print("ENCODER CONVOLUTIONAL PARAMETERS:")

    layercombo = load_specific_param(conv_layers, conv_params, "layercombo", '', '', default_conv_params)
    supercombo = recursive_concatcombo(layercombo) # turns list + tuples into all lists
    supercombo = ''.join(flatten_list(supercombo)) # flattens list recursively
    layercombine = load_specific_param(conv_layers, conv_params, "layercombine", '', '', default_conv_params)

    filters = load_specific_param(conv_layers, conv_params, "filters", 'c', supercombo, default_conv_params)
    conv_size = load_specific_param(conv_layers, conv_params, "conv_size", 'c', supercombo, default_conv_params)
    conv_strides = load_specific_param(conv_layers, conv_params, "conv_strides", 'c', supercombo, default_conv_params)
    padding = load_specific_param(conv_layers, conv_params, "padding", 'c', supercombo, default_conv_params)
    dilation_rate = load_specific_param(conv_layers, conv_params, "dilation_rate", 'c', supercombo, default_conv_params)
    pool_size = load_specific_param(conv_layers, conv_params, "pool_size", 'p', supercombo, default_conv_params)
    pool_strides = load_specific_param(conv_layers, conv_params, "pool_strides", 'p', supercombo, default_conv_params)
    pad_size = load_specific_param(conv_layers, conv_params, "pad_size", 'z', supercombo, default_conv_params)

    filters_up = load_specific_param(conv_layers, conv_params, "filters_up", 'u', supercombo, default_conv_params)
    upconv_size = load_specific_param(conv_layers, conv_params, "upconv_size", 'u', supercombo, default_conv_params)
    upconv_strides = load_specific_param(conv_layers, conv_params, "upconv_strides", 'u', supercombo, default_conv_params)
    upconv_type = load_specific_param(conv_layers, conv_params, "upconv_type", 'u', supercombo,default_conv_params)

    # batchnorm_pos = load_specific_param(conv_layers, default_conv_params, conv_params, "batchnorm_pos") #old version has batchnorm_bool
    if full_layers > 0:
        full_filters = load_specific_param(full_layers, conv_params, "full_filters", 'f', ['f'], default_conv_params, layer_str="fully connected") 
        dropout = load_specific_param(full_layers, conv_params, "dropout", 'f', ['f'], default_conv_params, layer_str="fully connected")
    else:
        full_filters = 0
        dropout = 0

    return filters, conv_size, conv_strides, padding, dilation_rate, pool_size, pool_strides, pad_size, filters_up, upconv_size, upconv_strides, upconv_type, layercombo, layercombine, full_filters, dropout


class Alex_Hyperopt_Encoder(Res_Encoder):
    def __init__(self, inputs, classes, weight_decay=0., weights=None, trainable=True, conv_layers=5, full_layers=2, conv_params=None):

        default_conv_params = {"filters": [96,256,384,384,256],
            "conv_size": [(7,7),(5,5),(3,3),(3,3),(3,3)],
            "conv_strides": [(1,1),(1,1),(1,1),(1,1),(1,1)],
            "padding": ['valid','valid','valid','valid','valid'],
            "dilation_rate": [(1,1),(1,1),(1,1),(1,1),(1,1)],
            "pool_size": [(2,2),(2,2),(1,1),(1,1),(2,2)],
            "pool_strides": [(2,2),(2,2),(1,1),(1,1),(2,2)],
            "pad_size": [(0,0),(0,0),(0,0),(0,0),(0,0)],
            "filters_up": [100],
            "upconv_size": [(2,2)],
            "upconv_strides": [(1,1)],
            "layercombo": ["capb","capb","capb","capb","capb"],
            "full_filters": [4096,4096],
            "dropout": [0.5,0.5]}
        filters, conv_size, conv_strides, padding, dilation_rate, pool_size, pool_strides, pad_size, filters_up, upconv_size, upconv_strides, layercombo, full_filters, dropout = \
            load_conv_params(conv_layers, full_layers, default_conv_params, conv_params)

        # actual start of CNN
        blocks = []
        for i in range(conv_layers):
            block_name = 'alexblock{}'.format(i + 1)
            block = alex_conv(filters[i], conv_size[i], conv_strides=conv_strides[i], padding=padding[i], pad_bool=True, pad_size=pad_size[i], pool_size=pool_size[i],
                pool_strides=pool_strides[i], dilation_rate=dilation_rate[i], filters_up=filters_up[i], kernel_size_up=upconv_size[i], strides_up=upconv_strides[i],
                layercombo=layercombo[i], weight_decay=weight_decay, block_name=block_name)
            blocks.append(block)

        for i in range(full_layers):
            block_name='alexfc{}'.format(i + 1)
            if i==0:
                block = alex_fc(full_filters[i], flatten_bool=True, dropout_bool=True, dropout=dropout[i], weight_decay=weight_decay, block_name=block_name)
            else:
                block = alex_fc(full_filters[i], flatten_bool=False, dropout_bool=True, dropout=dropout[i], weight_decay=weight_decay, block_name=block_name)
            blocks.append(block)

        super(Alex_Hyperopt_Encoder, self).__init__(inputs=inputs, blocks=blocks, weights=weights, trainable = trainable)

class Recursive_Hyperopt_Encoder(Res_Encoder):
    def __init__(self, inputs, classes, weight_decay=0., weights=None, trainable=True, 
        conv_layers=5, full_layers=2, conv_params=None):

        default_conv_params = {"filters": [64,128,256,512,512],
            "conv_size": [(3,3),(3,3),(3,3),(3,3),(3,3)],
            "conv_strides": [(1,1),(1,1),(1,1),(1,1),(1,1)],
            "padding": ['same','same','same','same','same'],
            "dilation_rate": [(1,1),(1,1),(1,1),(1,1),(1,1)],
            "pool_size": [(2,2),(2,2),(2,2),(2,2),(2,2)],
            "pool_strides": [(2,2),(2,2),(1,1),(1,1),(1,1)],
            "pad_size": [(0,0),(0,0),(0,0),(0,0),(0,0)],
            "filters_up": [100],
            "upconv_size": [(2,2)],
            "upconv_strides": [(1,1)],
            "layercombo": ["cacapb","cacapba","cacacapb","cacacapb","cacacapb"],
            "full_filters": [2048,2048],
            "dropout": [0.5,0.5]}
        filters, conv_size, conv_strides, padding, dilation_rate, pool_size, pool_strides, pad_size, filters_up, upconv_size, upconv_strides, upconv_type, layercombo, layercombine, full_filters, dropout = \
            load_conv_params(conv_layers, full_layers, default_conv_params, conv_params)

        # actual start of CNN
        blocks = []
        for i in range(conv_layers):
            if type(layercombine[i]) is list:
                combinecount = len(layercombine[i])-1
            else:
                combinecount = 0

            block_name = 'vgg_convblock{}'.format(i + 1)
            block = recursive_conv(filters[i], conv_size[i], conv_strides=conv_strides[i], padding=padding[i], pad_bool=False, pad_size=pad_size[i], pool_size=pool_size[i],
                    pool_strides=pool_strides[i], dilation_rate=dilation_rate[i], filters_up=filters_up[i], kernel_size_up=upconv_size[i], strides_up=upconv_strides[i], upconv_type=upconv_type[i],
                    layercombo=layercombo[i], layercombine=layercombine[i], combinecount=[-1], weight_decay=weight_decay, block_name=block_name)
            blocks.append(block)

        if full_layers > 0:
            block_name = 'vgg_fcblock'
            block = vgg_fcblock(full_filters, full_layers, dropout_bool=True, dropout=dropout, weight_decay=weight_decay, block_name=block_name)
            blocks.append(block)

        super(Recursive_Hyperopt_Encoder, self).__init__(inputs=inputs, blocks=blocks, weights=weights, trainable = trainable)

class VGGEncoder(Encoder):
    """VGG VGGEncoder.

    :param inputs: 4D Tensor, the input tensor
    :param filters: 1D array, number of filters per block
    :param convs: 1D array, number of convolutional layers per block, with
    length the same as `filters`.

    :return A Keras Model with outputs including the output of
    each block except `pool5` (using drop7 from `pool5` instead)

    >>> from keras_fcn.encoders import VGGEncoder
    >>> from keras.layers import Input
    >>> x = Input(shape=(224, 224, 3))
    >>> encoder = VGGEncoder(Input(x),
    >>>                  filters=[64, 128, 256, 512, 512],
    >>>                  convs=[2, 2, 3, 3, 3])
    >>> feat_pyramid = encoder.outputs

    """

    def __init__(self, inputs, filters, convs, weight_decay=0.,
            weights=None, trainable=True):
        blocks = []

        # Convolutional blocks
        for i, (fltr, conv) in enumerate(zip(filters, convs)):
            block_name = 'block{}'.format(i + 1)
            block = vgg_conv(filters=fltr, convs=conv, padding=False,
                             weight_decay=weight_decay,
                             block_name=block_name)
            blocks.append(block)

        # Fully Convolutional block
        fc_block = vgg_fc(filters=1024, weight_decay=weight_decay)
        # fc_block = vgg_fc(filters=4096, weight_decay=weight_decay)

        blocks.append(fc_block)

        super(VGGEncoder, self).__init__(inputs=inputs, blocks=blocks,
                                         weights=weights, trainable=trainable)


class VGG16(VGGEncoder):
    """A VGG16 feature encoder.

    >>> from keras_fcn.encoders import VGG16
    >>> from keras.layers import Input
    >>> x = Input(shape=(224, 224, 3))
    >>> encoder = VGG16(x)
    >>> feat_pyramid = encoder.outputs

    """

    def __init__(self, inputs, weight_decay=0.,
            weights='imagenet', trainable=True):
        if weights == 'imagenet':
            weights = 'https://github.com/fchollet/deep-learning-models/releases/download/v0.1/vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5'
        else:
            weights = None

        super(VGG16, self).__init__(inputs,
                                    filters=[64, 128, 196, 256, 384],
                                    convs=[2, 2, 3, 3, 3],
                                    weight_decay=weight_decay,
                                    weights=weights,
                                    trainable=trainable)

        # super(VGG16, self).__init__(inputs,
        #                             filters=[64, 128, 256, 512, 512],
        #                             convs=[2, 2, 3, 3, 3],
        #                             weight_decay=weight_decay,
        #                             weights=weights,
        #                             trainable=trainable)


class VGG19(VGGEncoder):
    """VGG19 net.

    >>> from keras_fcn.encoders import VGG19
    >>> from keras.layers import Input
    >>> x = Input(shape=(224, 224, 3))
    >>> encoder = VGG19(x)
    >>> feat_pyramids = encoder.outputs

    """

    def __init__(self, inputs, weight_decay=0.,
            weights='imagenet', trainable=True):
        if weights == 'imagenet':
            weights = 'https://github.com/fchollet/deep-learning-models/releases/download/v0.1/vgg19_weights_tf_dim_ordering_tf_kernels_notop.h5'
        else:
            weights = None

        super(VGG19, self).__init__(inputs,
                                    filters=[64, 128, 256, 512, 512],
                                    convs=[2, 2, 4, 4, 4],
                                    weight_decay=weight_decay,
                                    weights=weights,
                                    trainable=trainable)
