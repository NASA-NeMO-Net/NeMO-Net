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
from NeMO_encoders import load_specific_param
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
    print("---------------------------------------------------------")
    print("DECODER DECONVOLUTIONAL PARAMETERS:")

    # convs = load_specific_param(deconv_layers, default_deconv_params, deconv_params, "convs", layer_str="deconvolutional")
    scales = load_specific_param(deconv_layers, default_deconv_params, deconv_params, "scales", layer_str="deconvolutional")
    filters = load_specific_param(deconv_layers, default_deconv_params, deconv_params, "filters", layer_str="deconvolutional")
    conv_size = load_specific_param(deconv_layers, default_deconv_params, deconv_params, "conv_size", layer_str="deconvolutional")
    layercombo = load_specific_param(deconv_layers, default_deconv_params, deconv_params, "layercombo", layer_str="deconvolutional")

    return scales, filters, conv_size, layercombo


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
    for k in deconv_params:
        if len(deconv_params[k]) != len(pyramid)-1:
            print("Error: Deconvolution parameter {} is not the same length as pyramid-1".format(k))
            raise ValueError
    # remember that pyramid must have 1 extra, for target_shape purposes
    for p in pyramid:
        p_filters.append(p.shape[0])

    default_deconv_params = {"scales": [1,1,1,1,1],
        "filters": p_filters[:-1],
        "conv_size": [(2,2),(2,2),(2,2),(2,2),(2,2)],
        "layercombo": ["cacab","cacab","cacab","cacab","cacab"]}

    scales, filters, conv_size, layercombo = load_deconv_params(len(pyramid)-1, default_deconv_params, deconv_params)

    blocks = []
    for i in range(len(pyramid)-1):
        block_name = 'vgg_deconvblock{}'.format(i+1)
        block = vgg_deconvblock(filters[i], conv_size[i], classes, layercombo=layercombo[i], target_shape=K.int_shape(pyramid[i+1]), 
            scale=scales[i], weight_decay=weight_decay, block_name=block_name)
        blocks.append(block)

    return Decoder(pyramid=pyramid[:-1], blocks=blocks)
