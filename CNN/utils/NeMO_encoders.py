from __future__ import (
    absolute_import,
    unicode_literals
)
import numpy as np
import keras
import keras.backend as K

from keras.models import Model
from keras.utils.data_utils import get_file
from keras.utils import layer_utils

from keras.layers import Cropping2D
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
    def __init__(self, inputs, blocks, crop_shapes=None, weights=None,
                 trainable=True, name='encoder'):
        inverse_pyramid = []

        split_inputs = None 
        if crop_shapes is not None:             # used for multi-input models
            split_inputs = []
            for shape in crop_shapes:
                crop_len = (int(inputs.shape[1])-shape[0])/2
                if (int(inputs.shape[1]) - shape[0])%2 == 0:
                    crop_len = int(crop_len)
                    split_inputs.append(Cropping2D(cropping=((crop_len,crop_len), (crop_len,crop_len)))(inputs))
                else:
                    crop_len_lo = int(np.trunc(crop_len))
                    crop_len_hi = int(np.ceil(crop_len))
                    split_inputs.append(Cropping2D(cropping=((crop_len_lo,crop_len_hi), (crop_len_lo,crop_len_hi)))(inputs))

        # all parallel blocks
        if split_inputs is not None:
            inputs_copy = list(split_inputs)
        else:
            inputs_copy = inputs

        for i, block in enumerate(blocks):
            if i == 0:
                x = block(inputs_copy)
            else:
                x = block(x)
            if type(x) is list:
                inverse_pyramid.append(list(x))
            else:
                inverse_pyramid.append(x)

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

def load_specific_param(num_layers, default_conv_params, conv_params, specific_param, layer_str="convolutional"):
    default_param = default_conv_params[specific_param]
    try:
        param = conv_params[specific_param]
        if len(param) != num_layers:
            print("Found {} {} layers but {} {}, will only replace initial {} {}...".format(num_layers, layer_str, len(param), specific_param, len(param), specific_param))
            for i in range(len(param), num_layers):
                param.append(default_param[i])
        print("{}: {}".format(specific_param, param[:num_layers]))
    except:
        print("{} not found, reverting to default: {}".format(specific_param,default_param[:num_layers]))
        param = default_param

    return param

def load_conv_params(conv_layers, full_layers, default_conv_params, conv_params):
    print("---------------------------------------------------------")
    print("ENCODER CONVOLUTIONAL PARAMETERS:")

    filters = load_specific_param(conv_layers, default_conv_params, conv_params, "filters")
    conv_size = load_specific_param(conv_layers, default_conv_params, conv_params, "conv_size")
    padding = load_specific_param(conv_layers, default_conv_params, conv_params, "padding")
    dilation_rate = load_specific_param(conv_layers, default_conv_params, conv_params, "dilation_rate")
    pool_size = load_specific_param(conv_layers, default_conv_params, conv_params, "pool_size")
    pad_size = load_specific_param(conv_layers, default_conv_params, conv_params, "pad_size")
    layercombo = load_specific_param(conv_layers, default_conv_params, conv_params, "layercombo")
    # batchnorm_pos = load_specific_param(conv_layers, default_conv_params, conv_params, "batchnorm_pos") #old version has batchnorm_bool
    full_filters = load_specific_param(full_layers, default_conv_params, conv_params, "full_filters", layer_str="fully connected")
    dropout = load_specific_param(full_layers, default_conv_params, conv_params, "dropout", layer_str="fully connected")

    return filters, conv_size, padding, dilation_rate, pool_size, pad_size, layercombo, full_filters, dropout


class Alex_Hyperopt_Encoder(Res_Encoder):
    def __init__(self, inputs, classes, weight_decay=0., weights=None, trainable=True, conv_layers=5, full_layers=2, conv_params=None):

        default_conv_params = {"filters": [96,256,384,384,256],
            "conv_size": [(7,7),(5,5),(3,3),(3,3),(3,3)],
            "padding": ['valid','valid','valid','valid','valid'],
            "dilation_rate": [(1,1),(1,1),(1,1),(1,1),(1,1)],
            "pool_size": [(2,2),(2,2),(1,1),(1,1),(2,2)],
            "pad_size": [(0,0),(0,0),(0,0),(0,0),(0,0)],
            "layercombo": ["capb","capb","capb","capb","capb"],
            "full_filters": [4096,4096],
            "dropout": [0.5,0.5]}
        filters, conv_size, padding, dilation_rate, pool_size, pad_size, layercombo, full_filters, dropout = \
            load_conv_params(conv_layers, full_layers, default_conv_params, conv_params)

        # actual start of CNN
        blocks = []
        for i in range(conv_layers):
            block_name = 'alexblock{}'.format(i + 1)
            block = alex_conv(filters[i], conv_size[i], conv_strides=(1,1), padding=padding[i], pad_bool=True, pad_size=pad_size[i], pool_size=pool_size[i],
                pool_strides=pool_size[i], dilation_rate=dilation_rate[i], layercombo=layercombo[i], weight_decay=weight_decay, block_name=block_name)
            blocks.append(block)

        for i in range(full_layers):
            block_name='alexfc{}'.format(i + 1)
            if i==0:
                block = alex_fc(full_filters[i], flatten_bool=True, dropout_bool=True, dropout=dropout[i], weight_decay=weight_decay, block_name=block_name)
            else:
                block = alex_fc(full_filters[i], flatten_bool=False, dropout_bool=True, dropout=dropout[i], weight_decay=weight_decay, block_name=block_name)
            blocks.append(block)

        super(Alex_Hyperopt_Encoder, self).__init__(inputs=inputs, blocks=blocks, weights=weights, trainable = trainable)

class Alex_Parallel_Hyperopt_Encoder(Res_Encoder):
    def __init__(self, inputs, classes, weight_decay=0., weights=None, trainable=True, conv_layers=3, full_layers=2, conv_params=None):

        default_conv_params = {"filters": [64,128,256,384,512],
            "conv_size": [(7,7),(3,3),(3,3),(3,3),(3,3)],
            "padding": ['valid','valid','valid','valid','valid'],
            "dilation_rate": [(1,1),(1,1),(1,1),(1,1),(1,1)],
            "pool_size": [(2,2),(2,2),(1,1),(1,1),(1,1)],
            "pad_size": [(0,0),(0,0),(0,0),(0,0),(0,0)],
            "batchnorm_bool": [True,True,True,True,True],
            "full_filters": [4096,2048],
            "dropout": [0.5,0.5]}
        filters, conv_size, padding, dilation_rate, pool_size, pad_size, batchnorm_bool, full_filters, dropout = \
            load_conv_params(conv_layers, full_layers, default_conv_params, conv_params)

            # actual start of CNN
        blocks = []
        for i in range(conv_layers):
            block_name = 'parallel_block{}'.format(i + 1)
            block = parallel_conv(filters[i], conv_size[i], padding=padding[i], pad_size=pad_size[i], pool_size=pool_size[i], dilation_rate=dilation_rate[i],
                batchnorm_bool=batchnorm_bool[i], weight_decay=weight_decay, block_name=block_name)
            blocks.append(block)

        block_name = 'poolconcat_block'
        block = pool_concat(pool_size=(1,1), batchnorm_bool=True, block_name=block_name)
        blocks.append(block)

        for i in range(full_layers):
            block_name='alexfc{}'.format(i + 1)
            if i==0:
                block = alex_fc(full_filters[i], flatten_bool=True, dropout_bool=True, dropout=dropout[i], weight_decay=weight_decay, block_name=block_name)
            else:
                block = alex_fc(full_filters[i], flatten_bool=False, dropout_bool=True, dropout=dropout[i], weight_decay=weight_decay, block_name=block_name)
            blocks.append(block)

        super(Alex_Parallel_Hyperopt_Encoder, self).__init__(inputs=inputs, blocks=blocks, crop_shapes=None, weights=weights, trainable = trainable)

class VGG_Hyperopt_Encoder(Res_Encoder):
    def __init__(self, inputs, classes, weight_decay=0., weights=None, trainable=True, 
        conv_layers=5, full_layers=2, conv_params=None):

        default_conv_params = {"filters": [64,128,256,512,1024],
            "conv_size": [(3,3),(3,3),(3,3),(3,3),(3,3)],
            "padding": ['same','same','same','same','same'],
            "dilation_rate": [(1,1),(1,1),(1,1),(1,1),(1,1)],
            "pool_size": [(2,2),(2,2),(2,2),(2,2),(2,2)],
            "pad_size": [(0,0),(0,0),(0,0),(0,0),(0,0)],
            "layercombo": ["cacapb","cacapba","cacacapb","cacacapb","cacacapb"],
            "full_filters": [2048,2048],
            "dropout": [0.5,0.5]}
        filters, conv_size, padding, dilation_rate, pool_size, pad_size, layercombo, full_filters, dropout = \
            load_conv_params(conv_layers, full_layers, default_conv_params, conv_params)

        # actual start of CNN
        blocks = []
        for i in range(conv_layers):
            block_name = 'vgg_convblock{}'.format(i + 1)
            block = alex_conv(filters[i], conv_size[i], conv_strides=(1,1), padding=padding[i], pad_bool=False, pad_size=pad_size[i], pool_size=pool_size[i],
                pool_strides=pool_size[i], dilation_rate=dilation_rate[i], layercombo=layercombo[i], weight_decay=weight_decay, block_name=block_name)
            # block = vgg_convblock(filters[i], conv_size[i], convs=convs[i], padding=padding[i], batchnorm_bool=batchnorm_bool[i], pad_size=pad_size[i], 
            #     pool_size=pool_size[i], dilation_rate=dilation_rate[i], layercombo = layercombo[i], weight_decay=weight_decay, block_name=block_name)
            blocks.append(block)

        if full_layers > 0:
            block_name = 'vgg_fcblock'
            block = vgg_fcblock(full_filters, full_layers, dropout_bool=True, dropout=dropout, weight_decay=weight_decay, block_name=block_name)
            blocks.append(block)

        super(VGG_Hyperopt_Encoder, self).__init__(inputs=inputs, blocks=blocks, crop_shapes=None, weights=weights, trainable = trainable)

class Res34_Encoder(Res_Encoder):
    def __init__(self, inputs, classes, weight_decay=0., weights=None, trainable=True, fcflag = False):
        weights = None
        filters = [64, 128, 256, 512]
        convs = [2,2,2,2]
        reps = [3,4,6,3]
        blocks = []

        init_block = res_initialconv(filters=64, weight_decay = weight_decay)
        blocks.append(init_block)

        for i, (fltr, conv) in enumerate(zip(filters, convs)):
            block_name = 'megablock{}'.format(i + 1)
            if i==0:
                block = res_megaconv(fltr, conv, reps[i], init_strides=(1,1), weight_decay=weight_decay, block_name=block_name)
            else:
                block = res_megaconv(fltr, conv, reps[i], init_strides=(2,2), weight_decay=weight_decay, block_name=block_name)
            blocks.append(block)

        if fcflag:
            fc_block = res_fc(classes=classes, weight_decay=weight_decay)
        else:
            fc_block = res_1b1conv(filters=512, convs=1, weight_decay = weight_decay)
        blocks.append(fc_block)

        super(Res34_Encoder, self).__init__(inputs=inputs, blocks=blocks, weights=weights, trainable=trainable)

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
