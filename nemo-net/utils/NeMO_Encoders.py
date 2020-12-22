from typing import Tuple, Callable, List, Union, Dict

import numpy as np
import tensorflow as tf
import keras
import keras.backend as K

#from keras.models import Model
from keras.engine.training import Model
from keras.utils.data_utils import get_file
from keras.utils import layer_utils

from keras.layers import Cropping2D, Concatenate, Add
from NeMO_Blocks import (
    recursive_conv,
    dense_block
)
from NeMO_Backend import load_weights


class Res_Encoder(Model):
    """ Encoder structure for Fully Convolutional Networks

    inputs: Tensor input
    blocks: List of callable CNN blocks
    traininable: Trainable layers
    name: Name of encoder
    """
    def __init__(self, 
        inputs: tf.Tensor, 
        blocks: List[Callable], 
        trainable: bool = True, 
        name: str = 'encoder'):

        inverse_pyramid = []

        # If multiple inputs
        if type(inputs) is list:
            inputs_copy = [np.copy(inp) for inp in inputs]
        else:
            inputs_copy = inputs

        # start passing input tensor through blocks
        for i, block in enumerate(blocks):
            if i == 0:
                x = block(inputs_copy)
            else:
                x = block(x)

            # Keep track of block outputs as they get deeper
            if type(x) is list: # Output of block is list (multiple outputs)
                inverse_pyramid.append(list(x))
            else:
                inverse_pyramid.append(x)

        # reverse block outputs so that they go from deepest -> shallowest
        pyramid = reversed(inverse_pyramid)
        outputs = []
        for item in pyramid:
            if type(item) is list:
                for miniitem in item:
                    outputs.append(miniitem)
            else:
                outputs.append(item)

        # Creates encoder model with relevant inputs and outputs. 
        # Note that outputs is a list composed from convolutional blocks from deepest -> shallowest
        super(Res_Encoder, self).__init__(inputs = inputs, outputs = outputs)

        # Freezing basenet weights
        if trainable is False:
            for layer in self.layers:
                if layer.name in layer_names:
                    layer.trainable = False

def recursive_concatcombo(layercombo: Union[str, Tuple, List]) -> List:
    ''' Recursive function to take layercombo, which is organized through tuples and lists to represent parallel and feedforward connections,
        into a large list of strings for each segment of layers. Note that there can exist lists within lists in the output.

        layercombo: List that represents ENTIRE CNN architecture
    '''
    if type(layercombo) is str:
        return layercombo

    if type(layercombo) is tuple or list:
        a = [recursive_concatcombo(combo) for combo in layercombo]
    return a

def flatten_list(S: List) -> List[str]:
    ''' Recursive function that potential can take embedded lists of strings, and outputs one long string of all inputs

        S: Embedded list of strings (usually recursive_concatcombo(layercombo))
    '''
    if S == []:
        return S
    if isinstance(S[0], list):
        return flatten_list(S[0]) + flatten_list(S[1:])
    return S[:1] + flatten_list(S[1:])

def load_specific_param(num_blocks: int, 
    conv_params: Dict[str, List], 
    specific_param: str, 
    combokey: str, 
    supercombo: str,
    block_str: str = "convolutional") -> List:
    ''' Helper function to load a specific parameter from conv_params, given that it is a Dictionary of Lists.
        Takes into account if it's only 1 element (used for entire CNN), or has same # of elements as num_blocks

        num_blocks: # of major convolutional blocks in the CNN
        conv_params: Convolutional parameters in dictionary format
        specific_param: specific parameter to load (all of them are loaded similarly EXCEPT for 'layercombo' and 'layercombine')
        combokey: Specific key that indicates a certain type of layer (e.g. 'c' for conv2D, 'p' for pooling, etc...) that current param is tied to
        supercombo: The entire str (depth-first) across all blocks in 'layercombo'. This is used to check if a certain parameter exists
            within the architecture. If it doesn't, then there is no need to load it. You can also pass a string that includes all the unique
            combokeys that exist in the CNN.
        block_str: name of block
    '''

    param = []

    if specific_param is "layercombo": # parameter to load is layercombo (which determines architecture of CNN)
        try: # load the layercombo list from dictionary
            param = conv_params[specific_param]
        except:
            print("Please specify {} parameter of length {}".format(specific_param, num_blocks))

        if len(param) == 1:     # if only one param, use that for all the layers
            param = param * num_blocks

        if len(param) != num_blocks:
            print("{} parameter not the same length as the # of {} blocks: {}".format(specific_param, block_str, num_blocks))
            raise ValueError
    elif specific_param is "layercombine": # parameter to load is layercombine (which determines how to combine parallel connections)
        try:
            param = conv_params[specific_param]
        except:
            print("{} not specified... will use 'sum' function where appropriate".format(specific_param))
            param = ['sum'] * num_blocks
        
        if len(param) == 1:
            param = param * num_blocks
            
        if len(param) != num_blocks:
            print("{} parameter not the same length as the # of {} blocks: {}".format(specific_param, block_str, num_blocks))
            raise ValueError
    else:
        if combokey in supercombo:
            try:
                param = conv_params[specific_param]
            except:
                print("{} parameter not found, please specify it in the params dictionary as it is required".format(specific_param))
                raise ValueError

            if len(param) == 1:     # if only one param, use that for all the layers
                param = param * num_blocks

            if len(param) != num_blocks:
                print("{} parameter not the same length as the # of {} blocks: {}".format(specific_param, block_str, num_blocks))
                raise ValueError
    if len(param) is 0:
        param = [None] * num_blocks

    print("{}: {}".format(specific_param, param))

    return param

def load_conv_params(conv_blocks: int, 
    dense_blocks: int, 
    conv_params: Dict[str, List]) -> Tuple:
    ''' Helper function to load convolutional params

        conv_blocks: # of convolutional blocks
        dense_blocks: # of dense blocks (after convolutional blocks)
        conv_params: Convolutional parameters
    '''

    print("---------------------------------------------------------")
    print("ENCODER CONVOLUTIONAL PARAMETERS:")

    layercombo = load_specific_param(conv_blocks, conv_params, "layercombo", '', '')
    supercombo = recursive_concatcombo(layercombo) # turns list + tuples of strings into all embedded list of strings
    supercombo = ''.join(flatten_list(supercombo)) # flattens list recursively (if there are lists within lists). 
    layercombine = load_specific_param(conv_blocks, conv_params, "layercombine", '', '')

    filters = load_specific_param(conv_blocks, conv_params, "filters", 'c', supercombo)
    conv_size = load_specific_param(conv_blocks, conv_params, "conv_size", 'c', supercombo)
    conv_strides = load_specific_param(conv_blocks, conv_params, "conv_strides", 'c', supercombo)
    padding = load_specific_param(conv_blocks, conv_params, "padding", 'c', supercombo)
    dilation_rate = load_specific_param(conv_blocks, conv_params, "dilation_rate", 'c', supercombo)
    scaling = load_specific_param(conv_blocks, conv_params, "scaling", 's', supercombo)
    pool_size = load_specific_param(conv_blocks, conv_params, "pool_size", 'p', supercombo)
    pool_strides = load_specific_param(conv_blocks, conv_params, "pool_strides", 'p', supercombo)
    pad_size = load_specific_param(conv_blocks, conv_params, "pad_size", 'z', supercombo)

    filters_up = load_specific_param(conv_blocks, conv_params, "filters_up", 'u', supercombo)
    upconv_size = load_specific_param(conv_blocks, conv_params, "upconv_size", 'u', supercombo)
    upconv_strides = load_specific_param(conv_blocks, conv_params, "upconv_strides", 'u', supercombo)
    upconv_type = load_specific_param(conv_blocks, conv_params, "upconv_type", 'u', supercombo)

    dropout = load_specific_param(conv_blocks, conv_params, "dropout", 'd', supercombo)

    if dense_blocks > 0:
        dense_filters = load_specific_param(dense_blocks, conv_params, "dense_filters", 'f', ['f'], block_str="dense") 
        dense_dropout = load_specific_param(dense_blocks, conv_params, "dense_dropout", 'f', ['f'], block_str="dense")
    else:
        dense_filters = 0
        dense_dropout = 0

    return filters, \
        conv_size, \
        conv_strides, \
        padding, \
        dilation_rate, \
        scaling, \
        pool_size, \
        pool_strides, \
        pad_size, \
        filters_up, \
        upconv_size, \
        upconv_strides, \
        upconv_type,\
        dropout, \
        layercombo, \
        layercombine, \
        dense_filters, \
        dense_dropout


class Recursive_Encoder(Res_Encoder):
    def __init__(self, 
        inputs: tf.Tensor, 
        classes: int, 
        weight_decay: float = 0., 
        trainable: bool = True, 
        conv_blocks: int = 5, 
        dense_blocks: int = 2, 
        conv_params: Dict[str, List] = None):
        ''' Recursive Encoder that creates CNN structure from convolutional blocks through parallel and feedforward connections

            inputs: Tensor input
            classes: # of total classes
            weight_decay: weight decay
            trainable: If entire structure is trainable
            conv_blocks: # of major convolutional blocks
            dense_blocks: # of major dense layers
            conv_params: Convolutional parameters
        '''

        # default convolution parameters, consisting of 5 convolutional blocks in a VGG-like structure
        # Note that the upconv parameters don't have any bearing on the architecture, since no upsampling layers were used
        default_conv_params = {"filters": [64,128,256,512,512],
            "conv_size": [(3,3),(3,3),(3,3),(3,3),(3,3)],
            "conv_strides": [(1,1),(1,1),(1,1),(1,1),(1,1)],
            "padding": ['same','same','same','same','same'],
            "dilation_rate": [(1,1),(1,1),(1,1),(1,1),(1,1)],
            "scaling": [1,1,1,1,1],
            "pool_size": [(2,2),(2,2),(2,2),(2,2),(2,2)],
            "pool_strides": [(2,2),(2,2),(1,1),(1,1),(1,1)],
            "pad_size": [(0,0),(0,0),(0,0),(0,0),(0,0)],
            "filters_up": [None],
            "upconv_size": [None],
            "upconv_strides": [None],
            "dropout": [0.0],
            "layercombo": ["cacapb","cacapba","cacacapb","cacacapb","cacacapb"],
            "dense_filters": [2048,2048],
            "dense_dropout": [0.5,0.5]}
        
        loaded_conv_params = load_conv_params(conv_blocks, dense_blocks, conv_params)

        filters = loaded_conv_params[0]
        conv_size = loaded_conv_params[1]
        conv_strides = loaded_conv_params[2]
        padding = loaded_conv_params[3]
        dilation_rate = loaded_conv_params[4]
        scaling = loaded_conv_params[5]
        pool_size = loaded_conv_params[6]
        pool_strides = loaded_conv_params[7]
        pad_size = loaded_conv_params[8]
        filters_up = loaded_conv_params[9]
        upconv_size = loaded_conv_params[10]
        upconv_strides = loaded_conv_params[11]
        upconv_type = loaded_conv_params[12]
        dropout = loaded_conv_params[13]
        layercombo = loaded_conv_params[14]
        layercombine = loaded_conv_params[15]
        dense_filters = loaded_conv_params[16]
        dense_dropout = loaded_conv_params[17]
            
        # Start of CNN by blocks
        blocks = []
        for i in range(conv_blocks):
            # Number of times parallel layers will combine
            if type(layercombine[i]) is list:
                combinecount = len(layercombine[i])-1
            else:
                combinecount = 0

            block_name = 'NeMO_convblock{}'.format(i + 1)
            block = recursive_conv(filters[i], 
                conv_size[i], 
                conv_strides = conv_strides[i], 
                padding = padding[i], 
                pad_bool = False,
                pad_size = pad_size[i], 
                pool_size = pool_size[i],
                pool_strides = pool_strides[i], 
                dilation_rate = dilation_rate[i], 
                scaling = scaling[i], 
                filters_up = filters_up[i], 
                kernel_size_up = upconv_size[i], 
                strides_up = upconv_strides[i], 
                upconv_type = upconv_type[i],
                dropout = dropout[i],
                layercombo = layercombo[i], 
                layercombine = layercombine[i], 
                combinecount = [-1], 
                weight_decay = weight_decay, 
                block_name = block_name)
            blocks.append(block)

        if dense_blocks > 0:
            for i in range(dense_blocks):
                block_name = 'NeMO_denseblock{}'.format(i+1)
                if i == 0:
                    block = dense_block(dense_filters[i], 
                        dropout = dense_dropout[i],
                         weight_decay = weight_decay, 
                         block_name = block_name, 
                         do_flatten = True)
                else:
                    block = dense_block(dense_filters[i], 
                        dropout = dense_dropout[i], 
                        weight_decay = weight_decay, 
                        block_name = block_name, 
                        do_flatten = False)
                blocks.append(block)

        super(Recursive_Encoder, self).__init__(inputs = inputs, blocks = blocks, trainable = trainable)