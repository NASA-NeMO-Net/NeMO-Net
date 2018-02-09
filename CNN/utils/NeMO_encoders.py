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

        # all blocks
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

def load_conv_params(conv_layers, full_layers, default_conv_params, conv_params):
    default_filters = default_conv_params["filters"]
    try:
        filters = conv_params["filters"]
        if len(filters) != conv_layers:
            print("Found %d convolutional layers but %d filters, will only replace initial %d filters..." %(conv_layers,len(filters),len(filters)))
            for i in range(len(filters), conv_layers):
                filters.append(default_filters[i])
        print("filters: ", filters[:conv_layers])
    except:
        print("filters not found, reverting to default: ", default_filters[:conv_layers])
        filters = default_filters

    default_conv_size = default_conv_params["conv_size"]
    try:
        conv_size = conv_params["conv_size"]
        if len(conv_size) != conv_layers:
            print("Found %d convolutional layers but %d conv_size, will only replace initial %d conv_size..." %(conv_layers,len(conv_size),len(conv_size)))
            for i in range(len(conv_size), conv_layers):
                conv_size.append(default_conv_size[i])
        print("conv_size: ", conv_size[:conv_layers])
    except:
        print("conv_size not found, reverting to default: ", default_conv_size[:conv_layers])
        conv_size = default_conv_size

    default_dilation_rate = default_conv_params["dilation_rate"]
    try:
        dilation_rate = conv_params["dilation_rate"]
        if len(dilation_rate) != conv_layers:
            print("Found %d convolutional layers but %d dilation_rate, will only replace initial %d dilation_rate..." %(conv_layers,len(dilation_rate),len(dilation_rate)))
            for i in range(len(dilation_rate), conv_layers):
                dilation_rate.append(default_dilation_rate[i])
        print("dilation_rate: ", dilation_rate[:conv_layers])
    except:
        print("dilation_rate not found, reverting to default: ", default_dilation_rate[:conv_layers])
        dilation_rate = default_dilation_rate

    default_pool_size = default_conv_params["pool_size"]
    try:
        pool_size = conv_params["pool_size"]
        if len(pool_size) != conv_layers:
            print("Found %d convolutional layers but %d pool_size, will only replace initial %d pool_size..." %(conv_layers,len(pool_size),len(pool_size)))
            for i in range(len(pool_size), conv_layers):
                pool_size.append(default_pool_size[i])
        print("pool_size: ", pool_size[:conv_layers])
    except:
        print("pool_size not found, reverting to default: ", default_pool_size[:conv_layers])
        pool_size = default_pool_size
    pool_stride = pool_size     # usually true

    default_pad_size = default_conv_params["pad_size"]
    try:
        pad_size  = conv_params["pad_size"]
        if len(pad_size) != conv_layers:
            print("Found %d convolutional layers but %d pad_size, will only replace initial %d pad_size..." %(conv_layers,len(pad_size),len(pad_size)))
            for i in range(len(pad_size), conv_layers):
                pad_size.append(default_pad_size[i])
        print("pad_size: ", pad_size[:conv_layers])
    except:
        print("pad_size not found, reverting to default: ", default_pad_size[:conv_layers])
        pad_size = default_pad_size

    default_batchnorm_bool = default_conv_params["batchnorm_bool"]
    try:
        batchnorm_bool  = conv_params["batchnorm_bool"]
        if len(batchnorm_bool) != conv_layers:
            print("Found %d convolutional layers but %d batchnorm_bool, will only replace initial %d batchnorm_bool..." %(conv_layers,len(batchnorm_bool),len(batchnorm_bool)))
            for i in range(len(batchnorm_bool), conv_layers):
                batchnorm_bool.append(default_batchnorm_bool[i])
        print("batchnorm_bool: ", batchnorm_bool[:conv_layers])
    except:
        print("batchnorm_bool not found, reverting to default: ", default_batchnorm_bool[:conv_layers])
        batchnorm_bool = default_batchnorm_bool

    default_full_filters = default_conv_params["full_filters"]
    try:
        full_filters  = conv_params["full_filters"]
        if len(full_filters) != full_layers:
            print("Found %d fully connected layers but %d full_filters, will only replace initial %d full_filters..." %(full_layers,len(full_filters),len(full_filters)))
            for i in range(len(full_filters), full_layers):
                full_filters.append(default_full_filters[i])
        print("full_filters: ", full_filters[:full_layers])
    except:
        print("full_filters not found, reverting to default: ", default_full_filters[:full_layers])
        full_filters = default_full_filters


    default_dropout = default_conv_params["dropout"]
    try:
        dropout  = conv_params["dropout"]
        if len(dropout) != full_layers:
            print("Found %d fully connected layers but %d dropout, will only replace initial %d dropout..." %(full_layers,len(dropout),len(dropout)))
            for i in range(len(dropout), full_layers):
                dropout.append(default_dropout[i])
        print("dropout: ", dropout[:full_layers])
    except:
        print("dropout not found, reverting to default: ", default_dropout[:full_layers])
        dropout = default_dropout

    return filters, conv_size, dilation_rate, pool_size, pad_size, batchnorm_bool, full_filters, dropout


class Alex_Hyperopt_Encoder(Res_Encoder):
    def __init__(self, inputs, classes, weight_decay=0., weights=None, trainable=True, conv_layers=5, full_layers=2, conv_params=None):
        default_filters = [96, 256, 384, 384, 256]
        try:
            filters = conv_params["filters"]
            if len(filters) != conv_layers:
                print("Found %d convolutional layers but %d filters, will only replace initial %d filters..." %(conv_layers,len(filters),len(filters)))
                for i in range(len(filters), conv_layers):
                    filters.append(default_filters[i])
            print("filters: ", filters[:conv_layers])
        except:
            print("filters not found, reverting to default: ", default_filters[:conv_layers])
            filters = default_filters

        default_conv_size = [(7,7),(5,5),(3,3),(3,3),(3,3)]
        try:
            conv_size = conv_params["conv_size"]
            if len(conv_size) != conv_layers:
                print("Found %d convolutional layers but %d conv_size, will only replace initial %d conv_size..." %(conv_layers,len(conv_size),len(conv_size)))
                for i in range(len(conv_size), conv_layers):
                    conv_size.append(default_conv_size[i])
            print("conv_size: ", conv_size[:conv_layers])
        except:
            print("conv_size not found, reverting to default: ", default_conv_size[:conv_layers])
            conv_size = default_conv_size

        default_dilation_rate = [(1,1),(1,1),(1,1),(1,1),(1,1)]
        try:
            dilation_rate = conv_params["dilation_rate"]
            if len(dilation_rate) != conv_layers:
                print("Found %d convolutional layers but %d dilation_rate, will only replace initial %d dilation_rate..." %(conv_layers,len(dilation_rate),len(dilation_rate)))
                for i in range(len(dilation_rate), conv_layers):
                    dilation_rate.append(default_dilation_rate[i])
            print("dilation_rate: ", dilation_rate[:conv_layers])
        except:
            print("dilation_rate not found, reverting to default: ", default_dilation_rate[:conv_layers])
            dilation_rate = default_dilation_rate

        default_pool_size = [(2,2),(2,2),(1,1),(1,1),(2,2)]
        try:
            pool_size = conv_params["pool_size"]
            if len(pool_size) != conv_layers:
                print("Found %d convolutional layers but %d pool_size, will only replace initial %d pool_size..." %(conv_layers,len(pool_size),len(pool_size)))
                for i in range(len(pool_size), conv_layers):
                    pool_size.append(default_pool_size[i])
            print("pool_size: ", pool_size[:conv_layers])
        except:
            print("pool_size not found, reverting to default: ", default_pool_size[:conv_layers])
            pool_size = default_pool_size
        pool_stride = pool_size     # usually true


        default_pad_size = [(0,0),(0,0),(0,0),(0,0),(0,0)]
        try:
            pad_size  = conv_params["pad_size"]
            if len(pad_size) != conv_layers:
                print("Found %d convolutional layers but %d pad_size, will only replace initial %d pad_size..." %(conv_layers,len(pad_size),len(pad_size)))
                for i in range(len(pad_size), conv_layers):
                    pad_size.append(default_pad_size[i])
            print("pad_size: ", pad_size[:conv_layers])
        except:
            print("pad_size not found, reverting to default: ", default_pad_size[:conv_layers])
            pad_size = default_pad_size

        default_batchnorm_bool = [True, True, False, False, False]
        try:
            batchnorm_bool  = conv_params["batchnorm_bool"]
            if len(batchnorm_bool) != conv_layers:
                print("Found %d convolutional layers but %d batchnorm_bool, will only replace initial %d batchnorm_bool..." %(conv_layers,len(batchnorm_bool),len(batchnorm_bool)))
                for i in range(len(batchnorm_bool), conv_layers):
                    batchnorm_bool.append(default_batchnorm_bool[i])
            print("batchnorm_bool: ", batchnorm_bool[:conv_layers])
        except:
            print("batchnorm_bool not found, reverting to default: ", default_batchnorm_bool[:conv_layers])
            batchnorm_bool = default_batchnorm_bool

        default_full_filters = [4096, 4096]
        try:
            full_filters  = conv_params["full_filters"]
            if len(full_filters) != full_layers:
                print("Found %d fully connected layers but %d full_filters, will only replace initial %d full_filters..." %(full_layers,len(full_filters),len(full_filters)))
                for i in range(len(full_filters), full_layers):
                    full_filters.append(default_full_filters[i])
            print("full_filters: ", full_filters[:full_layers])
        except:
            print("full_filters not found, reverting to default: ", default_full_filters[:full_layers])
            full_filters = default_full_filters


        default_dropout = [0.5, 0.5]
        try:
            dropout  = conv_params["dropout"]
            if len(dropout) != full_layers:
                print("Found %d fully connected layers but %d dropout, will only replace initial %d dropout..." %(full_layers,len(dropout),len(dropout)))
                for i in range(len(dropout), full_layers):
                    dropout.append(default_dropout[i])
            print("dropout: ", dropout[:full_layers])
        except:
            print("dropout not found, reverting to default: ", default_dropout[:full_layers])
            dropout = default_dropout


        # actual start of CNN
        blocks = []
        for i in range(conv_layers):
            block_name = 'alexblock{}'.format(i + 1)
            block = alex_conv(filters[i], conv_size[i], pad_bool=True, pool_bool=True, batchnorm_bool=batchnorm_bool[i], pad_size=pad_size[i],
                pool_size=pool_size[i], pool_strides=pool_stride[i], dilation_rate=dilation_rate[i], weight_decay=weight_decay, block_name=block_name)
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
    def __init__(self, inputs, crop_shapes, classes, weight_decay=0., weights=None, trainable=True, conv_layers=3, full_layers=2, conv_params=None):

        default_conv_params = {"filters": [64,128,256,384,512],
            "conv_size": [(7,7),(3,3),(3,3),(3,3),(3,3)],
            "dilation_rate": [(1,1),(1,1),(1,1),(1,1),(1,1)],
            "pool_size": [(2,2),(2,2),(1,1),(1,1),(1,1)],
            "pad_size": [(0,0),(0,0),(0,0),(0,0),(0,0)],
            "batchnorm_bool": [True,True,True,True,True],
            "full_filters": [4096,2048],
            "dropout": [0.5,0.5]}
        filters, conv_size, dilation_rate, pool_size, pad_size, batchnorm_bool, full_filters, dropout = load_conv_params(conv_layers, full_layers, default_conv_params, conv_params)

            # actual start of CNN
        blocks = []
        for i in range(conv_layers):
            block_name = 'parallel_block{}'.format(i + 1)
            block = parallel_conv(filters[i], conv_size[i], pad_size=pad_size[i], pool_size=pool_size[i], dilation_rate=dilation_rate[i],
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

        super(Alex_Parallel_Hyperopt_Encoder, self).__init__(inputs=inputs, blocks=blocks, crop_shapes=crop_shapes, weights=weights, trainable = trainable)

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
                                    filters=[64, 128, 256, 384, 512],
                                    convs=[5, 4, 3, 3, 3],
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
