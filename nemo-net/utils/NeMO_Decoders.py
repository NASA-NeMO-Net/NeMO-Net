from typing import Tuple, Callable, List, Union, Dict

import tensorflow as tf
import keras
import keras.backend as K
from keras.models import Model
from keras.layers import (
    Input
)
from keras.layers.convolutional import (
    Conv2D
)
from NeMO_Layers import CroppingLike2D
from NeMO_Encoders import load_specific_param, flatten_list, recursive_concatcombo
from NeMO_Blocks import NeMO_Deconvblock


def Decoder(pyramid: List[tf.Tensor], blocks: List[Callable]):
    """A Functional decoder for decoder section of CNN

    :param: pyramid: A list of features in pyramid, scaling from large
                    receptive field to small receptive field.
    :param: blocks: List of callable CNN blocks
    """
    if len(blocks) != len(pyramid):
        raise ValueError('blocks needs to match the length of input pyramid')

    decoded = None
    # feat is from encoder, blocks from previous decoder block
    for feat, blk in zip(pyramid, blocks):
        decoded = blk(feat, decoded)

    return decoded

def load_deconv_params(deconv_blocks: int, 
    deconv_params: Dict[str, List], 
    block_name: str = "") -> Tuple:
    ''' Helper function to load deconvolutional params. Very similar to load_conv_params, except some parameters are assumed to be default values

        deconv_blocks: # of deconvolutional blocks
        deconv_params: Deconvolutional parameters
        block_str: Name of deconvolutional block
    '''


    print("---------------------------------------------------------")
    print("DECODER {} DECONVOLUTIONAL PARAMETERS:".format(block_name))

    layercombo = load_specific_param(deconv_blocks, deconv_params, "layercombo", "", "", block_str="deconvolutional")
    supercombo = recursive_concatcombo(layercombo) # turns list + tuples into all lists
    supercombo = ''.join(flatten_list(supercombo)) # flattens list recursively
    layercombine = load_specific_param(deconv_blocks, deconv_params, "layercombine", '', '')

    filters = load_specific_param(deconv_blocks, deconv_params, "filters", 'c', supercombo, block_str="deconvolutional")
    conv_size = load_specific_param(deconv_blocks, deconv_params, "conv_size", 'c', supercombo, block_str="deconvolutional")
    # Assumes conv_strides = 1
    # Assumes padding = same
    # Assumes dilation_rate = 1
    # Assumes dropout = 0
    pool_size = load_specific_param(deconv_blocks, deconv_params, "pool_size", "p", supercombo, block_str="deconvolutional")
    # Assumes pool_strides = 1
    # Assumes no padding layers (no pad_size)

    filters_up = load_specific_param(deconv_blocks, deconv_params, "filters_up", 'u', supercombo, block_str="deconvolutional")
    upconv_size = load_specific_param(deconv_blocks, deconv_params, "upconv_size", 'u', supercombo, block_str="deconvolutional")
    upconv_strides = load_specific_param(deconv_blocks, deconv_params, "upconv_strides", 'u', supercombo, block_str="deconvolutional")
    upconv_type = load_specific_param(deconv_blocks, deconv_params, "upconv_type", 'u', supercombo, block_str="deconvolutional")

    return filters, \
        conv_size, \
        pool_size, \
        filters_up, \
        upconv_size, \
        upconv_strides, \
        upconv_type, \
        layercombo, \
        layercombine


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
def NeMO_Decoder(pyramid: List, 
    classes: int, 
    scales: List[float], 
    weight_decay: float = 0., 
    bridge_params: Dict[str, List] = None, 
    prev_params: Dict[str, List] = None, 
    next_params: Dict[str, List] = None):
    ''' Functional Decoder that takes inputs from encoder outputs, and upsamples according to prev_block + bridge_block -> next_params

    pyramid: pyramid of outputs coming from encoder side of CNN (starting deep to shallow)
    classes: # of classes (if convolution to original # of classes is required)
    scales: Scales to multiply incoming connections from convolution branch by
    weight_decay: weight decay
    bridge_params: params that go into the bridge section (from output of convolution section of CNN to addition)
    prev_params: params that take previous output of deconv branch up to addition
    next_params: parmas that take addition and feed to next portion of CNN

    # Note prev_params and next_params can be combined, if determined to be same across all deconv structures... usually the initial block tends to be different
    '''

    # Size of convolutional filters coming from encoder side. Only calculated here for default parameter purposes, which is not currently used
    p_filters=[]
    for p in pyramid:
        p_filters.append(p.shape[0])

    if bridge_params is not None:
        default_bridge_params = {"filters": p_filters[:-1],
            "conv_size": [(2,2),(2,2),(2,2),(2,2),(2,2)],
            "filters_up": p_filters[:-1],
            "upconv_size": [(2,2)],
            "upconv_strides": [(1,1)],
            "layercombo": ["cacab","cacab","cacab","cacab","cacab"]}
        loaded_bridge_params = load_deconv_params(len(scales), bridge_params, "BRIDGE")
    if prev_params is not None:
        default_prev_params = {"filters": p_filters[:-1],
            "conv_size": [(2,2),(2,2),(2,2),(2,2),(2,2)],
            "filters_up": p_filters[:-1],
            "upconv_size": [(2,2)],
            "upconv_strides": [(1,1)],
            "layercombo": ["cba","cba","cba","cba","cba"]}
        loaded_prev_params = load_deconv_params(len(scales), prev_params, "PREV")
    if next_params is not None:
        default_next_params = {"filters": p_filters[:-1],
            "conv_size": [(2,2),(2,2),(2,2),(2,2),(2,2)],
            "filters_up": p_filters[:-1],
            "upconv_size": [(2,2)],
            "upconv_strides": [(1,1)],
            "layercombo": ["ba","ba","ba","ba","ba"]}
        loaded_next_params = load_deconv_params(len(scales), next_params, "NEXT")

    blocks = []
    for i in range(len(scales)):
        block_name = 'NeMO_deconvblock{}'.format(i+1)
        if bridge_params is not None:
            tempbridgeparams = [loaded_bridge_params[0][i], \
                loaded_bridge_params[1][i], \
                loaded_bridge_params[2][i], \
                loaded_bridge_params[3][i], \
                loaded_bridge_params[4][i], \
                loaded_bridge_params[5][i], \
                loaded_bridge_params[6][i], \
                loaded_bridge_params[7][i], \
                loaded_bridge_params[8][i]]
        else:
            tempbridgeparams = None

        if prev_params is not None:
            tempprevparams = [loaded_prev_params[0][i], \
                loaded_prev_params[1][i], \
                loaded_prev_params[2][i], \
                loaded_prev_params[3][i], \
                loaded_prev_params[4][i], \
                loaded_prev_params[5][i], \
                loaded_prev_params[6][i], \
                loaded_prev_params[7][i], \
                loaded_prev_params[8][i]]        
        else:
            tempprevparams = None

        if next_params is not None:
            tempnextparams = [loaded_next_params[0][i], \
                loaded_next_params[1][i], \
                loaded_next_params[2][i], \
                loaded_next_params[3][i], \
                loaded_next_params[4][i], \
                loaded_next_params[5][i], \
                loaded_next_params[6][i], \
                loaded_next_params[7][i], \
                loaded_next_params[8][i]]            
        else:
            tempnextparams = None

        # Note that vgg_deconvblock does not use recursive_conv, and hence cannot accomodate parallel architectures
        block = NeMO_Deconvblock(scales[i], 
            tempbridgeparams, 
            tempprevparams, 
            tempnextparams,
            weight_decay = weight_decay, 
            block_name = block_name, 
            count=i+1)
        blocks.append(block)

    return Decoder(pyramid=pyramid[:len(scales)], blocks = blocks)
