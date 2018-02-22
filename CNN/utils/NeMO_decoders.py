import keras
import keras.backend as K

from keras.models import Model
from keras.layers import (
    Input
)

from keras.layers.convolutional import (
    Conv2D
)

from NeMO_layers import CroppingLike2D
from NeMO_blocks import (
    vgg_deconv,
    vgg_score,
    vgg_upsampling,
    vgg_deconvblock
)
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials

def Decoder(pyramid, blocks):
    """A Functional decoder.

    :param: pyramid: A list of features in pyramid, scaling from large
                    receptive field to small receptive field.
                    The bottom of the pyramid is the input image.
    :param: blocks: A list of functions that takes a feature from the feature
                    pyramid, x, applies upsampling and accumulate the result
                    from the top of the pyramid.
                    The first block should expect the accumulated result, y,
                    is None. For example, see keras_fcn.blocks.vgg_deconv
                    The last block should simply apply a cropping on y to match
                    the shape of x. For example, see keras_fcn.blocks.vgg_score
    """
    if len(blocks) != len(pyramid):
        raise ValueError('`blocks` needs to match the length of'
                         '`pyramid`.')
    # decoded feature
    decoded = None
    for feat, blk in zip(pyramid, blocks):
        decoded = blk(feat, decoded)

    return decoded

def load_deconv_params(deconv_layers, default_deconv_params, deconv_params):
    print("DECODER DECONVOLUTIONAL PARAMETERS:")

    default_convs = default_deconv_params["convs"]
    try:
        convs = deconv_params["convs"]
        if len(convs) != deconv_layers:
            print("Found %d deconvolutional layers but %d convs, will only replace initial %d convs..." %(deconv_layers, len(convs), len(convs)))
            for i in range(len(convs), deconv_layers):
                convs.append(default_convs[i])
        print("convs: ", convs[:deconv_layers])
    except:
        print("convs not found, reverting to default: ", default_convs[:deconv_layers])
        convs = default_convs

    default_scales = default_deconv_params["scales"]
    try:
        scales = deconv_params["scales"]
        if len(scales) != deconv_layers:
            print("Found %d deconvolutional layers but %d scales, will only replace initial %d scales..." %(deconv_layers, len(scales), len(scales)))
            for i in range(len(scales), deconv_layers):
                scales.append(default_scales[i])
        print("scales: ", scales[:deconv_layers])
    except:
        print("scales not found, reverting to default: ", default_scales[:deconv_layers])
        scales = default_scales

    default_filters = default_deconv_params["filters"]
    try:
        filters = deconv_params["filters"]
        if len(filters) != deconv_layers:
            print("Found %d deconvolutional layers but %d filters, will only replace initial %d filters..." %(deconv_layers, len(filters), len(filters)))
            for i in range(len(filters), deconv_layers):
                filters.append(default_filters[i])
        print("filters: ", filters[:deconv_layers])
    except:
        print("filters not found, reverting to default: ", default_filters[:deconv_layers])
        filters = default_filters

    default_conv_size = default_deconv_params["conv_size"]
    try:
        conv_size = deconv_params["conv_size"]
        if len(conv_size) != deconv_layers:
            print("Found %d deconvolutional layers but %d conv_size, will only replace initial %d conv_size..." %(deconv_layers,len(conv_size),len(conv_size)))
            for i in range(len(conv_size), deconv_layers):
                conv_size.append(default_conv_size[i])
        print("conv_size: ", conv_size[:deconv_layers])
    except:
        print("conv_size not found, reverting to default: ", default_conv_size[:deconv_layers])
        conv_size = default_conv_size

    default_batchnorm_bool = default_deconv_params["batchnorm_bool"]
    try:
        batchnorm_bool  = deconv_params["batchnorm_bool"]
        if len(batchnorm_bool) != deconv_layers:
            print("Found %d deconvolutional layers but %d batchnorm_bool, will only replace initial %d batchnorm_bool..." %(deconv_layers,len(batchnorm_bool),len(batchnorm_bool)))
            for i in range(len(batchnorm_bool), deconv_layers):
                batchnorm_bool.append(default_batchnorm_bool[i])
        print("batchnorm_bool: ", batchnorm_bool[:deconv_layers])
    except:
        print("batchnorm_bool not found, reverting to default: ", default_batchnorm_bool[:deconv_layers])
        batchnorm_bool = default_batchnorm_bool

    return convs, scales, filters, conv_size, batchnorm_bool


def VGGDecoder(pyramid, scales, classes):
    """(Deprecated) A Functional decoder for the VGG Nets.

    :param: pyramid: A list of features in pyramid, scaling from large
                    receptive field to small receptive field.
                    The bottom of the pyramid is the input image.
    :param: scales: A list of weights for each of the feature map in the
                    pyramid, sorted in the same order as the pyramid.
    :param: classes: Integer, number of classes.
    """
    if len(scales) != len(pyramid) - 1:
        raise ValueError('`scales` needs to match the length of'
                         '`pyramid` - 1.')
    blocks = []

    features = pyramid[:-1]
    for i in range(len(features)):
        block_name = 'feat{}'.format(i + 1)
        if i < len(features) - 1:
            block = vgg_deconv(classes=classes, scale=scales[i],
                               kernel_size=(4, 4), strides=(2, 2),
                               crop_offset='centered',
                               weight_decay=1e-3,
                               block_name=block_name)
        else:
            block = vgg_deconv(classes=classes, scale=scales[i],
                               kernel_size=(16, 16), strides=(8, 8),
                               crop_offset='centered',
                               weight_decay=1e-3,
                               block_name=block_name)
        blocks.append(block)

    # Crop the decoded feature to match the image
    blocks.append(vgg_score(crop_offset='centered'))

    return Decoder(pyramid=pyramid, blocks=blocks)


def VGGUpsampler(pyramid, scales, classes, weight_decay=0.):
    """A Functional upsampler for the VGG Nets.

    :param: pyramid: A list of features in pyramid, scaling from large
                    receptive field to small receptive field.
                    The bottom of the pyramid is the input image.
    :param: scales: A list of weights for each of the feature map in the
                    pyramid, sorted in the same order as the pyramid.
    :param: classes: Integer, number of classes.
    """
    if len(scales) != len(pyramid) - 1:
        raise ValueError('`scales` needs to match the length of'
                         '`pyramid` - 1.')
    blocks = []

    for i in range(len(pyramid) - 1):
        block_name = 'feat{}'.format(i + 1)
        block = vgg_upsampling(classes=classes,
                               target_shape=K.int_shape(pyramid[i + 1]),
                               scale=scales[i],
                               weight_decay=weight_decay,
                               block_name=block_name)
        blocks.append(block)

    return Decoder(pyramid=pyramid[:-1], blocks=blocks)

def VGG_DecoderBlock(pyramid, classes, weight_decay=0., deconv_params=None):

    p_filters=[]
    # remember that pyramid must have 1 extra, for target_shape purposes
    for p in pyramid:
        p_filters.append(p.shape[0])

    default_deconv_params = {"convs": [2,2,2,2,2],
        "scales": [1,1,1,1,1],
        "filters": p_filters[:-1],
        "conv_size": [(2,2),(2,2),(2,2),(2,2),(2,2)],
        "batchnorm_bool": [True,True,True,True,True]}

    convs, scales, filters, conv_size, batchnorm_bool = load_deconv_params(len(pyramid)-1, default_deconv_params, deconv_params)

    blocks = []
    for i in range(len(pyramid)-1):
        block_name = 'vgg_deconvblock{}'.format(i+1)
        block = vgg_deconvblock(filters[i], conv_size[i], convs[i], classes, batchnorm_bool=batchnorm_bool[i], target_shape=K.int_shape(pyramid[i+1]), 
            scale=scales[i], weight_decay=weight_decay, block_name=block_name)
        blocks.append(block)

    return Decoder(pyramid=pyramid[:-1], blocks=blocks)
