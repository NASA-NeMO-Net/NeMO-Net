import keras.backend as K
import numpy as np
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
from NeMO_layers import CroppingLike2D, BilinearUpSampling2D
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials

def h(inp,c_count):
  # print("input: ", inp)
  if type(inp) is list or type(inp) is tuple:
    if len(inp) == 0:
      # print('a return: ', inp)
      return inp
    
    if type(inp) is tuple and len(inp) == 1:
      return inp[0]

    if type(inp[0]) is list or type(inp[0]) is tuple:
      # print('b return: ', inp[c_count])
      return inp[c_count]
    else:
      # print('c return: ', inp)
      return inp
  else:
    # print('d return: ', inp)
    return inp

def h_f(inp, c_count):
    if type(inp) is list or type(inp) is tuple:
        if len(inp) == 0:
            return inp
        elif len(inp) == 1:
            return inp[0] 
        else:
            return inp[c_count]
    else:
        return inp

def recursive_conv(filters, kernel_size, conv_strides=(1,1), padding='valid', pad_bool=False, pad_size=(0,0),
  pool_size=(2,2), pool_strides=(2,2), dilation_rate=(1,1), filters_up=None, kernel_size_up=None, strides_up=None, upconv_type="bilinear", dropout=0, 
  layercombo='capb', layercombine='sum', combinecount=[-1], weight_decay=0., block_name='convblock'):
    def f(input):
      x = input
      g = lambda input,c_count: input[c_count] if type(input) is list else input

      if type(layercombo) is tuple:
        startx = x
        end_x = []
        for i in range(len(layercombo)):
          x = recursive_conv(h_f(filters,i), h(kernel_size,i), h(conv_strides,i), h(padding,i), h(pad_bool,i), h(pad_size,i), h(pool_size,i),
            h(pool_strides,i), h(dilation_rate,i), h(filters_up,i), h(kernel_size_up,i), h(strides_up,i), h_f(upconv_type,i), h(dropout,i), 
            layercombo[i], layercombine, combinecount, weight_decay, block_name='{}_par{}'.format(block_name, i+1))(startx)
          # tempcombinecount += 1
          end_x.append(x)
        combinecount[0] = combinecount[0]+1

        # Code for figuring out layercombine... not very efficient currently but works
        if type(layercombine) is list:
          if layercombine[combinecount[0]] is "cat":
            x = concatenate(end_x, axis=-1)
          elif layercombine[combinecount[0]] is "sum":
            x = add(end_x)
          else:
            print("Undefined layercombine!")
            raise ValueError
        else:
          if layercombine is "cat":
            x = concatenate(end_x, axis=-1)
          elif layercombine is "sum":
            x = add(end_x)
          else:
            print("Undefined layercombine!")
            raise ValueError

      elif type(layercombo) is list:
        for i in range(len(layercombo)):
          x = recursive_conv(h_f(filters,i), h(kernel_size,i), h(conv_strides,i), h(padding,i), h(pad_bool,i), h(pad_size,i), h(pool_size,i),
            h(pool_strides,i), h(dilation_rate,i), h(filters_up,i), h(kernel_size_up,i), h(strides_up,i), h_f(upconv_type,i), h(dropout,i), 
            layercombo[i], layercombine, combinecount, weight_decay, block_name='{}_str{}'.format(block_name, i+1))(x)
          # tempcombinecount += 1
      else:
        x = alex_conv(filters, kernel_size, conv_strides, padding, pad_bool, pad_size, pool_size, pool_strides, dilation_rate, filters_up, kernel_size_up, strides_up, upconv_type, dropout, 
          layercombo, weight_decay, block_name)(x)
      return x
    return f

# General multi-purpose convolution block used for all convolutions
# filters: # of filters [int]
# kernel_size: Size of convolutional kernel [(int,int)]
# conv_strides: Stride of convolution kernel [(int,int)]
# padding: Type of padding for convolution ['valid' or 'same']
# pad_bool: Custom padding of the tensor [bool]
# pad_size: Custom padding size for the tensor [(int,int)]
# pool_size: Max pooling size [(int,int)]
# pool_strides: Max pooling stride size [(int,int)], usually same as pool_size
# dilation_rate: Convolution kernel dilation rate [(int,int)]
# layercombo: Combination of layers: ['c': Convolution, 'a': Activation, 'p': Pooling, 'b': Batch Normalization]
# weight_decay: kernel regularizer l2 weight decay [float]
# block_name: Name of block [string]
def alex_conv(filters, kernel_size, conv_strides=(1,1), padding='valid', pad_bool=False, pad_size=(0,0), 
  pool_size=(2,2), pool_strides=(2,2), dilation_rate=(1,1), filters_up=None, kernel_size_up=None, strides_up=None, upconv_type='bilinear', dropout=0, 
  layercombo='capb', weight_decay=0., block_name='alexblock'):
    def f(input):
      x = input
      c_total = layercombo.count("c") # 2D convolution
      a_total = layercombo.count("a") # Activation (relu)
      p_total = layercombo.count("p") # Pool
      b_total = layercombo.count("b") # Batch norm
      d_total = layercombo.count("d") # Dropout
      z_total = layercombo.count("z") # zero padding
      s_total = layercombo.count("s") # residual shortcut connection
      u_total = layercombo.count("u") # upsample
      c_count=0
      a_count=0
      p_count=0
      b_count=0
      d_count=0
      z_count=0
      s_count=0
      u_count=0
      f = lambda input,c_count: input[c_count] if type(input) is list else input
      start_x = x

      for layer_char in layercombo:
        if layer_char == "z":
          x = ZeroPadding2D(padding=f(pad_size,z_count), name="{}_Padding{}".format(block_name,z_count+1))(x)
          z_count +=1

        if layer_char == "c":
          # if pad_bool:
          #   x = ZeroPadding2D(padding=pad_size)(x)
          #   test_size = dilation_rate[0]*(kernel_size[0]-1)+1   # have to make sure this size is smaller than x.size
          #   if test_size > x.shape[1]:
          #     temp_padsize = int(np.ceil((test_size-int(x.shape[1]))/2))
          #     x = ZeroPadding2D(padding=(temp_padsize,temp_padsize))(x)
          print("block:", block_name, "filters:", f(filters,c_count), "conv_size:", f(kernel_size,c_count), "conv_strides:", f(conv_strides,c_count), 
            "padding:", f(padding,c_count), "dilation_rate:", f(dilation_rate,c_count), "weight_decay:", weight_decay)

          x = Conv2D(f(filters,c_count), f(kernel_size,c_count), strides=f(conv_strides,c_count), padding=f(padding,c_count), dilation_rate=f(dilation_rate,c_count),
            kernel_initializer='he_normal', kernel_regularizer=l2(weight_decay), name='{}_conv{}'.format(block_name,c_count+1))(x)
          c_count +=1

        if layer_char == "p":
          # if pool_size[0] > x.shape[1]:
          #   temp_padsize = int(np.ceil((pool_size[0]-int(x.shape[1]))/2))
          #   x = ZeroPadding2D(padding=(temp_padsize,temp_padsize))(x)
          x = MaxPooling2D(pool_size=f(pool_size,p_count), padding='same', strides=pool_strides, name='{}_pool{}'.format(block_name,p_count+1))(x)
          p_count +=1

        if layer_char == "b":
          x = BatchNormalization(name='{}_BatchNorm{}'.format(block_name, b_count+1))(x)
          b_count +=1

        if layer_char == "a":
          x = Activation('relu', name='{}_Activ{}'.format(block_name, a_count+1))(x)
          a_count +=1
        if layer_char == "d":
          x = Dropout(f(dropout,d_count), name='{}_Dropout{}'.format(block_name, d_count+1))(x)
          d_count +=1
        if layer_char == "s":
          x = res_shortcut(start_x, x, weight_decay, block_name='{}_Shortcut{}'.format(block_name,s_count+1))
          start_x = x
          s_count +=1
        if layer_char == "u":
          print("block: ", block_name, "filters_up:", f(filters_up,u_count), "conv_size_up:", f(kernel_size_up,u_count), "strides_up:", f(strides_up,u_count), "type:", f(upconv_type,u_count))
          if f(upconv_type,u_count) == "bilinear":
              xsize = K.int_shape(x)
              xsize = [i for i in xsize]
              xsize[1] = int(xsize[1]*f(strides_up,u_count)[0])
              xsize[2] = int(xsize[2]*f(strides_up,u_count)[1])
              xsize = tuple(xsize)
              x = BilinearUpSampling2D(target_shape=xsize, name='{}_BiUp{}'.format(block_name, u_count+1))(x)
          elif f(upconv_type,u_count) == "nn":
              xsize = K.int_shape(x)
              xsize = [i for i in xsize]
              xsize[1] = int(xsize[1]*f(strides_up,u_count)[0])
              xsize[2] = int(xsize[2]*f(strides_up,u_count)[1])
              xsize = tuple(xsize)
              x = BilinearUpSampling2D(target_shape=xsize, method='nn', name='{}_NNUp{}'.format(block_name, u_count+1))(x)
          elif f(upconv_type,u_count) == "2dtranspose":
              print("size: ", K.int_shape(x))
              x = Conv2DTranspose(f(filters_up,u_count), f(kernel_size_up,u_count), strides=f(strides_up,u_count), padding='same', kernel_initializer='he_normal', 
                name='{}_convT{}'.format(block_name, u_count+1))(x)
          else:
              print("Undefined upsampling method!")
              raise ValueError
          u_count +=1
      return x
    return f

def recursive_conv_wparams(filters, kernel_size, conv_strides=(1,1), padding='valid', pad_bool=False, pad_size=(0,0),
  pool_size=(2,2), pool_strides=(1,1), dilation_rate=(1,1), filters_up=None, kernel_size_up=None, strides_up=None, upconv_type="bilinear", dropout=0, 
  layercombo='capb', layercombine='sum', combinecount=[-1], weight_decay=0., block_name='convblock'):
    def f(input):
      x = input
      g = lambda input,c_count: input[c_count] if type(input) is list else input

      if type(layercombo) is tuple:
        startx = x
        end_x = []
        for i in range(len(layercombo)):
          x = recursive_conv(h_f(filters,i), h(kernel_size,i), h(conv_strides,i), h(padding,i), h(pad_bool,i), h(pad_size,i), h(pool_size,i),
            h(pool_strides,i), h(dilation_rate,i), h(filters_up,i), h(kernel_size_up,i), h(strides_up,i), h_f(upconv_type,i), h(dropout,i), 
            layercombo[i], layercombine, combinecount, weight_decay, block_name='{}_par{}'.format(block_name, i+1))(startx)
          # tempcombinecount += 1
          end_x.append(x)
        combinecount[0] = combinecount[0]+1

        # Code for figuring out layercombine... not very efficient currently but works
        if type(layercombine) is list:
          if layercombine[combinecount[0]] is "cat":
            x = concatenate(end_x, axis=-1)
          elif layercombine[combinecount[0]] is "sum":
            x = add(end_x)
          else:
            print("Undefined layercombine!")
            raise ValueError
        else:
          if layercombine is "cat":
            x = concatenate(end_x, axis=-1)
          elif layercombine is "sum":
            x = add(end_x)
          else:
            print("Undefined layercombine!")
            raise ValueError

      elif type(layercombo) is list:
        for i in range(len(layercombo)):
          x = recursive_conv(h_f(filters,i), h(kernel_size,i), h(conv_strides,i), h(padding,i), h(pad_bool,i), h(pad_size,i), h(pool_size,i),
            h(pool_strides,i), h(dilation_rate,i), h(filters_up,i), h(kernel_size_up,i), h(strides_up,i), h_f(upconv_type,i), h(dropout,i), 
            layercombo[i], layercombine, combinecount, weight_decay, block_name='{}_str{}'.format(block_name, i+1))(x)
          # tempcombinecount += 1
      else:
        x = alex_conv(filters, kernel_size, conv_strides, padding, pad_bool, pad_size, pool_size, pool_strides, dilation_rate, filters_up, kernel_size_up, strides_up, upconv_type, dropout, 
          layercombo, weight_decay, block_name)(x)
      return x
    return f

def vgg_deconvblock(classes, scale, bridge_params=None, prev_params=None, next_params=None, weight_decay=0., block_name='vgg_deconvblock', count=0):
  # params in this order:
  # filters, conv_size, filters_up, upconv_size, upconv_strides, upconv_type, layercombo, layercombine
    def f(x, y):
        if bridge_params is not None:
#           print("Bridge params: ", bridge_params[0], bridge_params[1], bridge_params[2], bridge_params[3], bridge_params[4], bridge_params[5], bridge_params[6], bridge_params[7])        
          x = recursive_conv_wparams(filters=bridge_params[0], kernel_size=bridge_params[1], pool_size=bridge_params[2], padding='same', filters_up=bridge_params[3], kernel_size_up=bridge_params[4], strides_up=bridge_params[5],
            upconv_type=bridge_params[6], layercombo=bridge_params[7], layercombine=bridge_params[8], combinecount=[-1], weight_decay=weight_decay, block_name='{}_bridgeconv{}'.format(block_name,count))(x)
          # x = alex_conv(bridge_params[0], bridge_params[1], padding='same', pad_size=(0,0), dilation_rate=(1,1), filters_up=bridge_params[2], kernel_size_up=bridge_params[3], strides_up=bridge_params[4],
          #   upconv_type=bridge_params[5], layercombo=bridge_params[6], weight_decay=weight_decay, block_name='{}_bridgeconv{}'.format(block_name,count))(x)

        if y is not None:
          if prev_params is not None:
            y = recursive_conv_wparams(filters=prev_params[0], kernel_size=prev_params[1], pool_size=prev_params[2], padding='same', filters_up=prev_params[3], kernel_size_up=prev_params[4], strides_up=prev_params[5],
            upconv_type=prev_params[6], layercombo=prev_params[7], layercombine=prev_params[8], combinecount=[-1], weight_decay=weight_decay, block_name='{}_prevconv{}'.format(block_name,count))(y)
#             y = alex_conv(prev_params[0], prev_params[1], padding='same', pad_size=(0,0), dilation_rate=(1,1), filters_up=prev_params[2], kernel_size_up=prev_params[3], strides_up=prev_params[4],
#               upconv_type=prev_params[5], layercombo=prev_params[6], weight_decay=weight_decay, block_name='{}_prevconv{}'.format(block_name,count))(y)
          def scaling(xx, ss=1):
            return xx * ss
          scaled = Lambda(scaling, arguments={'ss': scale}, name='{}_scale'.format(block_name))(x)
          x = add([y, scaled])

        if next_params is not None:
          x = recursive_conv_wparams(filters=next_params[0], kernel_size=next_params[1], pool_size=next_params[2], padding='same', filters_up=next_params[3], kernel_size_up=next_params[4], strides_up=next_params[5],
            upconv_type=next_params[6], layercombo=next_params[7], layercombine=next_params[8], combinecount=[-1], weight_decay=weight_decay, block_name='{}_nextconv{}'.format(block_name,count))(x)
#           x = alex_conv(next_params[0], next_params[1], padding='same', pad_size=(0,0), dilation_rate=(1,1), filters_up=next_params[2], kernel_size_up=next_params[3], strides_up=next_params[4],
#             upconv_type=next_params[5], layercombo=next_params[6], weight_decay=weight_decay, block_name='{}_nextconv{}'.format(block_name,count))(x)
        return x
    return f

def vgg_fcblock(filters, dropout=0.5, weight_decay=0., block_name='vgg_fcblock', first_time=False):
    def f(input):
      x = input
      if first_time:
          x = Flatten()(x)

      x = Dropout(dropout)(x)
      x = Dense(filters, activation='relu', kernel_initializer='he_normal', kernel_regularizer=l2(weight_decay), name='{}_Dense'.format(block_name))(x)
#       x = BatchNormalization(name='{}_BatchNorm'.format(block_name))(x)
      return x
    return f

def alex_fc(filters, flatten_bool=False, dropout_bool=False, dropout=0.5, weight_decay=0., block_name='alexfc'):
    def f(input):
      x = input
      if flatten_bool:
        x = Flatten()(x)

      x = Dense(filters, kernel_initializer='he_normal', kernel_regularizer=l2(weight_decay), name='{}_dense'.format(block_name))(x)
      if dropout_bool:
        x = Dropout(dropout)(x)
      return x
    return f


def res_shortcut(input, residual, weight_decay=0, block_name='res_shortcut'):
    """ Adds a shortcut between input and residual block and merges them with "sum"
    """
    shortcut = input
    input_shape = K.int_shape(input)
    residual_shape = K.int_shape(residual)

    stride_width = int(round(input_shape[1] / residual_shape[1]))
    stride_height = int(round(input_shape[2] / residual_shape[2]))
    equal_channels = input_shape[3] == residual_shape[3]

    if stride_width > 1 or stride_height > 1 or not equal_channels:
      shortcut = Conv2D(filters=residual_shape[3], kernel_size=(1,1), strides=(stride_width,stride_height),
        padding="valid", kernel_initializer="he_normal", kernel_regularizer=l2(weight_decay), name='{}_conv'.format(block_name))(input)

    return add([shortcut, residual])


# ALL functions below are from the original VGG FCN code, but they are hard-coded in many ways (filters, # of layers, etc...)
# Only really kept here for posterity

def vgg_conv(filters, convs, padding=False, weight_decay=0., block_name='blockx'):
    """A VGG convolutional block for encoding.
    # NOTE: All kernels are 3x3 hard-coded!

    :param filters: Integer, number of filters per conv layer
    :param convs: Integer, number of conv layers in the block
    :param block_name: String, the name of the block, e.g., block1

    >>> from keras_fcn.blocks import vgg_conv
    >>> x = vgg_conv(filters=64, convs=2, block_name='block1')(x)

    """
    def f(x):
        for i in range(convs):
            if block_name == 'block1' and i == 0:
                if padding is True:
                    x = ZeroPadding2D(padding=(100, 100))(x)
                x = Conv2D(filters, (3, 3), activation='relu', padding='same',
                           kernel_initializer='he_normal',
                           kernel_regularizer=l2(weight_decay),
                           name='{}_conv{}'.format(block_name, int(i + 1)))(x)
            else:
                x = Conv2D(filters, (3, 3), activation='relu', padding='same',
                           kernel_initializer='he_normal',
                           kernel_regularizer=l2(weight_decay),
                           name='{}_conv{}'.format(block_name, int(i + 1)))(x)

        pool = MaxPooling2D((2, 2), strides=(2, 2), padding='same',
                            name='{}_pool'.format(block_name))(x)
        return pool
    return f


def vgg_fc(filters, weight_decay=0., block_name='block5'):
    """A fully convolutional block for encoding.

    :param filters: Integer, number of filters per fc layer

    >>> from keras_fcn.blocks import vgg_fc
    >>> x = vgg_fc(filters=4096)(x)

    """
    def f(x):
        fc6 = Conv2D(filters, kernel_size=(7, 7),
                     activation='relu', padding='same',
                     dilation_rate=(2, 2),
                     kernel_initializer='he_normal',
                     kernel_regularizer=l2(weight_decay),
                     name='{}_fc6'.format(block_name))(x)
        drop6 = Dropout(0.5)(fc6)
        fc7 = Conv2D(filters, kernel_size=(1, 1),
                     activation='relu', padding='same',
                     kernel_initializer='he_normal',
                     kernel_regularizer=l2(weight_decay),
                     name='{}_fc7'.format(block_name))(drop6)
        drop7 = Dropout(0.5)(fc7)
        return drop7
    return f

def vgg_deconv(classes, scale=1, kernel_size=(4, 4), strides=(2, 2),
               crop_offset='centered', weight_decay=0., block_name='featx'):
    """A VGG convolutional transpose block for decoding.

    :param classes: Integer, number of classes
    :param scale: Float, scale factor to the input feature, varing from 0 to 1
    :param kernel_size: Tuple, the kernel size for Conv2DTranspose layers
    :param strides: Tuple, the strides for Conv2DTranspose layers
    :param crop_offset: Tuple or "centered", the offset for cropping.
    The default is "centered", which crop the center of the feature map.

    >>> from keras_fcn.blocks import vgg_deconv
    >>> x = vgg_deconv(classes=21, scale=1e-2, block_name='feat2')(x)

    """
    def f(x, y):
        def scaling(xx, ss=1):
            return xx * ss
        scaled = Lambda(scaling, arguments={'ss': scale},
                        name='scale_{}'.format(block_name))(x)
        score = Conv2D(filters=classes, kernel_size=(1, 1),
                       activation='linear',
                       kernel_initializer='he_normal',
                       kernel_regularizer=l2(weight_decay),
                       name='score_{}'.format(block_name))(scaled)
        if y is None:
            upscore = Conv2DTranspose(filters=classes, kernel_size=kernel_size,
                                      strides=strides, padding='valid',
                                      kernel_initializer='he_normal',
                                      kernel_regularizer=l2(weight_decay),
                                      use_bias=False,
                                      name='upscore_{}'.format(block_name))(score)
        else:
            crop = CroppingLike2D(target_shape=K.int_shape(y),
                                  offset=crop_offset,
                                  name='crop_{}'.format(block_name))(score)
            merge = add([y, crop])
            upscore = Conv2DTranspose(filters=classes, kernel_size=kernel_size,
                                      strides=strides, padding='valid',
                                      kernel_initializer='he_normal',
                                      kernel_regularizer=l2(weight_decay),
                                      use_bias=False,
                                      name='upscore_{}'.format(block_name))(merge)
        return upscore
    return f


def vgg_upsampling(classes, target_shape=None, scale=1, weight_decay=0., block_name='featx'):
    """A VGG convolutional block with bilinear upsampling for decoding.

    :param classes: Integer, number of classes
    :param scale: Float, scale factor to the input feature, varing from 0 to 1
    :param target_shape: 4D Tuples with targe_height, target_width as
    the 2nd, 3rd elements if `channels_last` or as the 3rd, 4th elements if
    `channels_first`.

    >>> from keras_fcn.blocks import vgg_upsampling
    >>> feat1, feat2, feat3 = feat_pyramid[:3]
    >>> y = vgg_upsampling(classes=21, target_shape=(None, 14, 14, None),
    >>>                    scale=1, block_name='feat1')(feat1, None)
    >>> y = vgg_upsampling(classes=21, target_shape=(None, 28, 28, None),
    >>>                    scale=1e-2, block_name='feat2')(feat2, y)
    >>> y = vgg_upsampling(classes=21, target_shape=(None, 224, 224, None),
    >>>                    scale=1e-4, block_name='feat3')(feat3, y)

    """
    def f(x, y):
        score = Conv2D(filters=classes, kernel_size=(1, 1),
                       activation='linear',
                       padding='valid',
                       kernel_initializer='he_normal',
                       kernel_regularizer=l2(weight_decay),
                       name='score_{}'.format(block_name))(x)
        if y is not None:
            def scaling(xx, ss=1):
                return xx * ss
            scaled = Lambda(scaling, arguments={'ss': scale},
                            name='scale_{}'.format(block_name))(score)
            score = add([y, scaled])
        upscore = BilinearUpSampling2D(
            target_shape=target_shape,
            name='upscore_{}'.format(block_name))(score)
        return upscore
    return f


def vgg_score(crop_offset='centered'):
    """A helper block to crop the decoded feature.

    :param crop_offset: Tuple or "centered", the offset for cropping.
    The default is "centered", which crop the center of the feature map.

    >>> from keras_fcn.blocks import vgg_deconv
    >>> score = vgg_score(crop_offset='centered')(image, upscore)

    """
    def f(x, y):
        score = CroppingLike2D(target_shape=K.int_shape(
            x), offset=crop_offset, name='score')(y)
        return score
    return f
