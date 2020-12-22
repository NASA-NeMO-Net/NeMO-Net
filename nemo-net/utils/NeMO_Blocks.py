# NeMO_Blocks.py contains the base layers before calling NeMO_Layers and Keras Layers

from typing import Tuple, Callable, List, Union, Dict, Any
from collections.abc import Iterable
import numpy as np

import tensorflow as tf

import keras.backend as K
from keras.layers import (
    Dropout,
    Lambda,
    Activation,
    Dense,
    Flatten,
    concatenate,
)
from keras.layers.convolutional import (
    Conv2D,
    Conv2DTranspose,
    MaxPooling2D,
    AveragePooling2D,
    ZeroPadding2D
)
from keras.layers.merge import add
from keras.layers.normalization import BatchNormalization
from keras.regularizers import l2

from NeMO_Layers import CroppingLike2D, BilinearUpSampling2D, PixelShuffler

def h(inp: Union[Tuple, List, Tuple[int,int]], c_count: int):
    ''' Helper function to separate tuple parameter (e.g. conv_stride) from list (feedforward) or tuple (parallel)
        
        input: Input parameter
        c_count: count of input parameter to take (if list or tuple indicates a feedforward or paralle connection). This ultimately comes from counting
            the layercombo parameter
    '''
    if type(inp) is list or type(inp) is tuple: # feedforward or maybe parallel connection
        if len(inp) == 0: # empty
            return inp # return empty
    
        if type(inp) is tuple and len(inp) == 1: # Only one element in parallel connection, meaning that element is used for the rest of the sublayers
            return inp[0] # return that one element

        if type(inp[0]) is list or type(inp[0]) is tuple: # feedforward or parallel connections nested inside
            return inp[c_count] # return element based on c_count
        else: # we are at end of branch, the input is the tuple parameter that we return 
            return inp
    else: # not feedforward nor parallel connection
        return inp # just return same element

def h_f(inp: Union[Tuple, List, int], c_count: int):
    ''' Helper function to separate singular parameter (e.g. filters) from list (feedforward) or tuple (parallel). 
        Similar to h, but for non-tuple parameters.
        
        input: Input parameter
        c_count: count of input parameter to take (if list or tuple indicates a feedforward or paralle connection). This ultimately comes from counting
            the layercombo parameter
    '''
    if type(inp) is list or type(inp) is tuple: # feedforward or parallel connection
        if len(inp) == 0: # empty
            return inp # return empty
        elif len(inp) == 1: # Only one element, meaning that element is used for the rest of the sublayers
            return inp[0] # return that one element
        else: # we are end of branch, that input is the parameter that we return
            return inp[c_count]
    else: # not feedforward nor parallel connection
        return inp # just return same element

def recursive_conv(filters: Union[Tuple, List, int],
    kernel_size: Union[Tuple, List, Tuple[int, int]],
    conv_strides: Union[Tuple, List, Tuple[int,int]] = (1,1),
    padding: Union[Tuple, List, str] = 'valid',
    pad_bool: Union[Tuple, List, bool] = False,
    pad_size: Union[Tuple, List, Tuple[int, int]] = (0,0),
    pool_size: Union[Tuple, List, Tuple[int, int]] = (2,2), 
    pool_strides: Union[Tuple, List, Tuple[int, int]] = (2,2), 
    dilation_rate: Union[Tuple, List, Tuple[int, int]] = (1,1), 
    scaling: Union[Tuple, List, float] = 1.0, 
    filters_up: Union[Tuple, List, int] = None, 
    kernel_size_up: Union[Tuple, List, Tuple[int, int]] = None, 
    strides_up: Union[Tuple, List, Tuple[int, int]] = None, 
    upconv_type: Union[Tuple, List, str] = "bilinear", 
    dropout: Union[Tuple, List, float] = 0.0, 
    layercombo: Union[Tuple, List, str] = 'capb', 
    layercombine: Union[List, str] = 'sum', 
    combinecount: List[int] = [-1], 
    weight_decay: float = 0., 
    block_name: str = 'convblock') -> Callable:
    ''' Key recursive convolutional block that recursively takes inputs (both feedforward and parallel) and constructs a CONVOLUTIONAL BLOCK
        as specified by layercombo. This is accomplished by going into the layercombo string, and going through each element. If the element is
        a tuple, then it indicates a parallel connection. If it is a list, it indicates a feedforward link. At the base level, layercombo will be 
        composed of a string, indicating the type of layer(s) to put together. For the layer by layer implementation, please refer to baselayer_conv

        filters: # of filters [int]
        kernel_size: Size of convolutional kernel [(int, int)]
        conv_strides: Stride of convolution kernel [(int, int)]
        padding: Type of padding for convolution ['valid' or 'same']
        pad_bool: Custom padding of the tensor [bool]
        pad_size: Custom padding size for the tensor [(int,int)]
        pool_size: Max pooling size [(int,int)]
        pool_strides: Max pooling stride size [(int,int)], usually same as pool_size
        dilation_rate: Convolution kernel dilation rate [(int,int)]
        scaling: Scale layer through multiplication [float]
        filters_up: # of filters for deconvolution [int]
        kernel_size_up: Size of deconvolution kernel [(int, int)]
        strides_up: Stride of deconvolution kernel [(int, int)]
        upconv_type: Type of deconvolution/ upscaling ['bilinear' or 'nn' or 'pixel_shuffle' or '2dtranspose'] Note: 'nn' -> nearest neighbor
        dropout: rate of dropout for dropout layer [float]
        layercombo: Combination of layers: ['c': Convolution, 'a': Activation, 'p': Pooling, 'b': Batch Normalization, 'z': Zero padding,
                                            's': Scaling, 'u': upscaling/ deconvolution]. This is key in how the layers are organized (either by
                                            parallel or by feedforward).
        layercombine: Indicates how layers should be combined for PARALLEL connections. ["sum" or "cat"] Note 'cat' -> concatenate
            This variable organized in a depth-first way. Please refer to examples as to how this is utilized. 
            However, often, this will simply be either "sum" or "cat" since we want to combine all layers similarly for the entire block
            NOTE: This is an all or nothing specification: i.e. either indicate how EVERY parallel connection should be combined, or just specify
                one method to combine ALL parallel branches
        weight_decay: kernel regularizer l2 weight decay [float]
        block_name: Name of block [str]
    '''

    def f(input: tf.Tensor):
        x = input
        
        # If layercombo is a tuple, indicating a parallel connection
        if type(layercombo) is tuple:
            startx = x
            end_x = []
            for i in range(len(layercombo)):
                # Recursive self call with helper functions
                x = recursive_conv(h_f(filters,i), 
                    h(kernel_size,i), 
                    h(conv_strides,i), 
                    h(padding,i), 
                    h(pad_bool,i), 
                    h(pad_size,i), 
                    h(pool_size,i),
                    h(pool_strides,i), 
                    h(dilation_rate,i), 
                    h(scaling,i), 
                    h(filters_up,i), 
                    h(kernel_size_up,i), 
                    h(strides_up,i), 
                    h_f(upconv_type,i), 
                    h(dropout,i), 
                    layercombo[i], 
                    layercombine, 
                    combinecount, 
                    weight_decay, 
                    block_name = '{}_par{}'.format(block_name, i+1))(startx)
                end_x.append(x)
            combinecount[0] = combinecount[0]+1 # keep tracks of which combination method we are on

            # Determining how to combine parallel branches
            # If a list, indicating different methods of combination for ALL parallel layers. 
            # Must be same length as all # of parallel branches
            if type(layercombine) is list: 
                if layercombine[combinecount[0]] is "cat":
                    x = concatenate(end_x, axis=-1)
                elif layercombine[combinecount[0]] is "sum":
                    x = add(end_x)
                else:
                    print("Undefined layercombine!")
                    raise ValueError
            else: # "cat" or "sum", indicating same method of combination for ALL layers within the block
                if layercombine is "cat":
                    x = concatenate(end_x, axis=-1)
                elif layercombine is "sum":
                    x = add(end_x)
                else:
                    print("Undefined layercombine!")
                    raise ValueError

        # If layercombo is a list, indicating a feedforward connection
        elif type(layercombo) is list:
            for i in range(len(layercombo)):
                # Recursive self call with helper functions
                x = recursive_conv(h_f(filters,i), 
                    h(kernel_size,i), 
                    h(conv_strides,i), 
                    h(padding,i), 
                    h(pad_bool,i), 
                    h(pad_size,i), 
                    h(pool_size,i),
                    h(pool_strides,i), 
                    h(dilation_rate,i), 
                    h(scaling,i), 
                    h(filters_up,i), 
                    h(kernel_size_up,i), 
                    h(strides_up,i), 
                    h_f(upconv_type,i), 
                    h(dropout,i), 
                    layercombo[i], 
                    layercombine, 
                    combinecount, 
                    weight_decay, 
                    block_name = '{}_str{}'.format(block_name, i+1))(x)
        # if neither list or tuple, we are at end of recursive, and start layer processing and return output tensor
        else: 
            x = baselayer_conv(filters, 
                kernel_size, 
                conv_strides, 
                padding, 
                pad_bool, 
                pad_size, 
                pool_size, 
                pool_strides, 
                dilation_rate, 
                scaling, 
                filters_up, 
                kernel_size_up, 
                strides_up, 
                upconv_type, 
                dropout, 
                layercombo, 
                weight_decay, 
                block_name)(x)

        return x
    return f


def baselayer_conv(filters: Union[List, int],
    kernel_size: Union[List, Tuple[int, int]],
    conv_strides: Union[List, Tuple[int, int]],
    padding: Union[List, str] = 'valid', 
    pad_bool: Union[List, bool] = False, 
    pad_size: Union[List, Tuple[int, int]] = (0,0),
    pool_size: Union[List, Tuple[int, int]] = (2,2), 
    pool_strides: Union[List, Tuple[int, int]] = (2,2), 
    dilation_rate: Union[List, Tuple[int, int]] = (1,1), 
    scaling: Union[List, float] = 1.0, 
    filters_up: Union[List, int] = None, 
    kernel_size_up: Union[List, Tuple[int, int]] = None, 
    strides_up: Union[List, Tuple[int, int]] = None, 
    upconv_type: Union[List, str] = 'bilinear', 
    dropout: Union[List, float] = 0.0,
    layercombo: Union[List, str] = 'capb', 
    weight_decay: Union[List, float] = 0., 
    block_name: str = 'alexblock') -> Callable:
    ''' General multi-purpose convolution block used for all convolutions, specifying parameters of layers as described in layercombo
        
        filters: # of filters [int]
        kernel_size: Size of convolutional kernel [(int, int)]
        conv_strides: Stride of convolution kernel [(int, int)]
        padding: Type of padding for convolution ['valid' or 'same']
        pad_bool: Custom padding of the tensor [bool]
        pad_size: Custom padding size for the tensor [(int,int)]
        pool_size: Max pooling size [(int,int)]
        pool_strides: Max pooling stride size [(int,int)], usually same as pool_size
        dilation_rate: Convolution kernel dilation rate [(int,int)]
        scaling: Scale layer through multiplication [float]
        filters_up: # of filters for deconvolution [int]
        kernel_size_up: Size of deconvolution kernel [(int, int)]
        strides_up: Stride of deconvolution kernel [(int, int)]
        upconv_type: Type of deconvolution/ upscaling ['bilinear' or 'nn' or 'pixel_shuffle' or '2dtranspose'] Note: 'nn' -> nearest neighbor
        dropout: rate of dropout for dropout layer [float]
        layercombo: Combination of layers: ['c': Convolution, 'a': Activation, 'p': Pooling, 'b': Batch Normalization, 'z': Zero padding,
                                            's': Scaling, 'u': upscaling/ deconvolution]
        weight_decay: kernel regularizer l2 weight decay [float]
        block_name: Name of block [str]
    '''

    def func(input: tf.Tensor):
        x = input
        c_total = layercombo.count("c") # 2D convolution
        a_total = layercombo.count("a") # Activation (relu)
        p_total = layercombo.count("p") # Pool
        b_total = layercombo.count("b") # Batch norm
        d_total = layercombo.count("d") # Dropout
        z_total = layercombo.count("z") # zero padding
        s_total = layercombo.count("s") # residual shortcut connection
        u_total = layercombo.count("u") # upsample
        c_count = 0
        a_count = 0
        p_count = 0
        b_count = 0
        d_count = 0
        z_count = 0
        s_count = 0 # used to be shortcut, now is scaling
        u_count = 0

        # Helper function that returns input[c_count] if input is a list, otherwise input
        # This is useful when dealing with a feedforward structure:
        # e.g. [(4,4), (3,3), (2,2)] for conv_size
        # However, we sometimes may only have 1 element, which is meant to be taken as a constant for all appropriate layers in the block
        # Hence, this function will pass the c_count element, or just the element itself if used across all layers
        f = lambda input, c_count: input[c_count] if type(input) is list else input

        for layer_char in layercombo:
            if layer_char == "z": # Zero padding layer
                x = ZeroPadding2D(padding = f(pad_size, z_count), 
                    name="{}_Padding{}".format(block_name,z_count+1))(x)
                z_count +=1

            if layer_char == "c": # 2D convolution layer
                print("block:", block_name, 
                    "filters:", f(filters,c_count),
                    "conv_size:", f(kernel_size,c_count), 
                    "conv_strides:", f(conv_strides,c_count), 
                    "padding:", f(padding,c_count), 
                    "dilation_rate:", f(dilation_rate,c_count), 
                    "weight_decay:", weight_decay)

                x = Conv2D(f(filters, c_count), 
                    f(kernel_size, c_count), 
                    strides = f(conv_strides, c_count), 
                    padding = f(padding, c_count), 
                    dilation_rate = f(dilation_rate,c_count),
                    kernel_initializer = 'he_normal', 
                    kernel_regularizer = l2(weight_decay), 
                    name='{}_conv{}'.format(block_name, c_count+1))(x)
                c_count +=1

            if layer_char == "p": # Pooling layer
                x = MaxPooling2D(pool_size = f(pool_size, p_count), 
                    padding = 'same', 
                    strides = pool_strides, 
                    name = '{}_pool{}'.format(block_name,p_count+1))(x)
                p_count +=1

            if layer_char == "b": # Batch Normalization
                x = BatchNormalization(name = '{}_BatchNorm{}'.format(block_name, b_count+1))(x)
                b_count +=1

            if layer_char == "a": # Activation Layer
                x = Activation('relu', name = '{}_Activ{}'.format(block_name, a_count+1))(x)
                a_count +=1
        
            if layer_char == "d": # Dropout Layer
                x = Dropout(f(dropout, d_count), name = '{}_Dropout{}'.format(block_name, d_count+1))(x)
                d_count +=1
        
            if layer_char == "s": # Scaling layer
                def scalefunc(xx, ss=1):
                    return xx * ss    
                x = Lambda(scalefunc, arguments={'ss': f(scaling, s_count)}, name='{}_scale{}'.format(block_name, s_count+1))(x)
                s_count +=1

            if layer_char == "u": # Upsampling layer
                print("block: ", block_name, 
                    "filters_up:", f(filters_up, u_count), 
                    "conv_size_up:", f(kernel_size_up, u_count), 
                    "strides_up:", f(strides_up, u_count), 
                    "type:", f(upconv_type, u_count))
                if f(upconv_type, u_count) == "bilinear":  # Auto calculate target shape if bilinear upsampling
                    xsize = K.int_shape(x)
                    xsize = [i for i in xsize]
                    xsize[1] = int(xsize[1]*f(strides_up, u_count)[0])
                    xsize[2] = int(xsize[2]*f(strides_up, u_count)[1])
                    xsize = tuple(xsize)
                    x = BilinearUpSampling2D(target_shape=xsize, name='{}_BiUp{}'.format(block_name, u_count+1))(x)
                elif f(upconv_type, u_count) == "nn": # Auto calculate target shape if nearest neighbor
                    xsize = K.int_shape(x)
                    xsize = [i for i in xsize]
                    xsize[1] = int(xsize[1]*f(strides_up,u_count)[0])
                    xsize[2] = int(xsize[2]*f(strides_up,u_count)[1])
                    xsize = tuple(xsize)
                    x = BilinearUpSampling2D(target_shape=xsize, method='nn', name='{}_NNUp{}'.format(block_name, u_count+1))(x)
                elif f(upconv_type, u_count) == "pixel_shuffle":
                    x = PixelShuffler()(x) # defaults to (2,2) upscaling only
                elif f(upconv_type, u_count) == "2dtranspose":
                    x = Conv2DTranspose(f(filters_up, u_count), 
                        f(kernel_size_up, u_count), 
                        strides = f(strides_up, u_count), 
                        padding = 'same', 
                        kernel_initializer ='he_normal', 
                        name = '{}_convT{}'.format(block_name, u_count+1))(x)
                else:
                    print("Undefined upsampling method!")
                    raise ValueError
                u_count += 1
        return x
    return func

def NeMO_Deconvblock(scale: float = 1.0, 
    bridge_params: List = None, 
    prev_params: List = None, 
    next_params: List = None, 
    weight_decay: float = 0., 
    block_name: str = 'vgg_deconvblock', 
    count: int = 0) -> Callable:
    ''' Deconvolution block to aid in recursive convolution within the deconvolution branches.
        All params are in this order: filters, conv_size, pool_size, filters_up, upconv_size, upconv_type, layercombo, layercombine

        scale: Scaling to be done on bridge output before it is summed with the prev_block
        bridge_params: List of bridge block parameters
        prev_params: List of previous block parameters
        next_params: List of next block parameters
        weight_decay: Weight decay
        block_name: Block name
        count: Count to keep track of which block we are in going from deepest back up to shallowest
    '''

    def f(x: tf.Tensor, y: tf.Tensor):
        if bridge_params is not None:
            x = recursive_conv(filters = bridge_params[0], 
                kernel_size = bridge_params[1], 
                pool_size = bridge_params[2], 
                pool_strides = (1,1),
                padding = 'same', 
                filters_up = bridge_params[3], 
                kernel_size_up = bridge_params[4], 
                strides_up = bridge_params[5],
                upconv_type = bridge_params[6], 
                layercombo = bridge_params[7], 
                layercombine = bridge_params[8], 
                combinecount = [-1], 
                weight_decay = weight_decay, 
                block_name = '{}_bridgeconv{}'.format(block_name,count))(x)

        if y is not None: # this will be none if it is the first element in the decoder block
            if prev_params is not None:
                y = recursive_conv(filters = prev_params[0], 
                    kernel_size = prev_params[1], 
                    pool_size = prev_params[2],
                    pool_strides = (1,1), 
                    padding = 'same', 
                    filters_up = prev_params[3], 
                    kernel_size_up = prev_params[4], 
                    strides_up = prev_params[5],
                    upconv_type = prev_params[6], 
                    layercombo = prev_params[7], 
                    layercombine = prev_params[8], 
                    combinecount = [-1], 
                    weight_decay = weight_decay, 
                    block_name = '{}_prevconv{}'.format(block_name,count))(y)
            def scaling(xx, ss=1):
                return xx * ss
            scaled = Lambda(scaling, arguments={'ss': scale}, name='{}_scale'.format(block_name))(x)
            x = add([y, scaled])

        if next_params is not None:
            x = recursive_conv(filters = next_params[0], 
                kernel_size = next_params[1], 
                pool_size = next_params[2],
                pool_strides = (1,1), 
                padding = 'same', 
                filters_up = next_params[3], 
                kernel_size_up = next_params[4], 
                strides_up = next_params[5],
                upconv_type = next_params[6], 
                layercombo = next_params[7], 
                layercombine = next_params[8], 
                combinecount = [-1], 
                weight_decay = weight_decay, 
                block_name = '{}_nextconv{}'.format(block_name,count))(x)

        return x
    return f

def dense_block(filters: int, 
    dropout: float = 0.5, 
    weight_decay: float = 0., 
    block_name: str = 'vgg_fcblock', 
    do_flatten: bool = False):
    ''' Dense layer w/ dropout layer

        filters: # of filters
        dropout: dropout rate
        weight_decay: weight decay rate
        block_name: block name
        do_flatten: Flatten input before passing into dense layers (important if input layer is 3D tensor)
    '''

    def f(input: tf.Tensor):
        x = input
        if do_flatten:
            x = Flatten()(x)

        x = Dropout(dropout)(x)
        x = Dense(filters, 
            activation='relu', 
            kernel_initializer='he_normal', 
            kernel_regularizer=l2(weight_decay), 
            name='{}_Dense'.format(block_name))(x)
        return x
    return f
