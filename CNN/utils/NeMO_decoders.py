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
    filters = load_specific_param(deconv_layers, default_deconv_params, deconv_params, "filters", layer_str="deconvolutional")
    conv_size = load_specific_param(deconv_layers, default_deconv_params, deconv_params, "conv_size", layer_str="deconvolutional")
    layercombo = load_specific_param(deconv_layers, default_deconv_params, deconv_params, "layercombo", layer_str="deconvolutional")

    filters_up = load_specific_param(deconv_layers, default_deconv_params, deconv_params, "filters_up")
    upconv_size = load_specific_param(deconv_layers, default_deconv_params, deconv_params, "upconv_size")
    upconv_strides = load_specific_param(deconv_layers, default_deconv_params, deconv_params, "upconv_strides")

    return filters, conv_size, filters_up, upconv_size, upconv_strides, layercombo


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

# Function for deconv block
# pyramid: pyramid of outputs coming from convolution side of CNN (starting deep to shallow)
# classes: # of classes (if convolution to original # of classes is required)
# scales: Scales to multiply incoming connections from convolution branch by
# weight_decay: weight decay
# bridge_params: params that go into the bridge section (from output of convolution section of CNN to addition)
# prev_params: params that take previous output of deconv branch up to addition
# next_params: parmas that take addition and feed to next portion of CNN
# upsample: Upsample end of each decoder block or not
# Note prev_params and next_params can be combined, if determined to be same across all deconv structures... usually the first one tends to be different
def VGG_DecoderBlock(pyramid, classes, scales, weight_decay=0., bridge_params=None, prev_params=None, next_params=None, upsample=True):

    p_filters=[]

    # These tests fail for 2 different situations: one for len(pyramid) and one for len(pyramid)+1
    # for k in bridge_params:
    #     if len(bridge_params[k]) != len(pyramid):
    #         print("Error: Deconvolution bridge parameter {} is not the same length as pyramid".format(k))
    #         raise ValueError
    # if len(scales) != len(pyramid):
    #     print("Error: scales parameter not the same length as pyramid")
    #     raise ValueError

    # remember that pyramid must have 1 extra, for target_shape purposes
    for p in pyramid:
        p_filters.append(p.shape[0])

    if bridge_params is not None:
        default_bridge_params = {"filters": p_filters[:-1],
            "conv_size": [(2,2),(2,2),(2,2),(2,2),(2,2)],
            "filters_up": p_filters[:-1],
            "upconv_size": [(2,2)],
            "upconv_strides": [(1,1)],
            "layercombo": ["cacab","cacab","cacab","cacab","cacab"]}
        bridge_filters, bridge_conv_size, bridge_filters_up, bridge_upconv_size, bridge_upconv_strides, bridge_layercombo = load_deconv_params(len(scales), default_bridge_params, bridge_params)
    if prev_params is not None:
        default_prev_params = {"filters": p_filters[:-1],
            "conv_size": [(2,2),(2,2),(2,2),(2,2),(2,2)],
            "filters_up": p_filters[:-1],
            "upconv_size": [(2,2)],
            "upconv_strides": [(1,1)],
            "layercombo": ["cba","cba","cba","cba","cba"]}
        prev_filters, prev_conv_size, prev_filters_up, prev_upconv_size, prev_upconv_strides, prev_layercombo = load_deconv_params(len(scales), default_prev_params, prev_params)
    if next_params is not None:
        default_next_params = {"filters": p_filters[:-1],
            "conv_size": [(2,2),(2,2),(2,2),(2,2),(2,2)],
            "filters_up": p_filters[:-1],
            "upconv_size": [(2,2)],
            "upconv_strides": [(1,1)],
            "layercombo": ["ba","ba","ba","ba","ba"]}
        next_filters, next_conv_size, next_filters_up, next_upconv_size, next_upconv_strides, next_layercombo = load_deconv_params(len(scales), default_next_params, next_params)

    blocks = []
    for i in range(len(scales)):
        block_name = 'vgg_deconvblock{}'.format(i+1)
        if bridge_params is not None:
            tempbridgeparams = [bridge_filters[i], bridge_conv_size[i], bridge_filters_up[i], bridge_upconv_size[i], bridge_upconv_strides[i], bridge_layercombo[i]]
        else:
            tempbridgeparams = None

        if prev_params is not None:
            tempprevparams = [prev_filters[i], prev_conv_size[i], prev_filters_up[i], prev_upconv_size[i], prev_upconv_strides[i], prev_layercombo[i]]
        else:
            tempprevparams = None

        if next_params is not None:
            tempnextparams = [next_filters[i], next_conv_size[i], next_filters_up[i], next_upconv_size[i], next_upconv_strides[i], next_layercombo[i]]
        else:
            tempnextparams = None

        # Note that vgg_deconvblock does not use recursive_conv, and hence cannot accomodate parallel architectures
        block = vgg_deconvblock(classes, scales[i], tempbridgeparams, tempprevparams, tempnextparams, upsample=upsample[i], target_shape=K.int_shape(pyramid[i+1]),
            weight_decay=weight_decay, block_name=block_name)
        blocks.append(block)

    return Decoder(pyramid=pyramid[:len(scales)], blocks=blocks)
