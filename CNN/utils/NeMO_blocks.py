import keras.backend as K
import numpy as np
from keras.layers import (
    Dropout,
    Lambda,
    Activation,
    Dense,
    Flatten,
    concatenate
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

    if type(inp[0]) is list or type(inp[0]) is tuple:
      # print('b return: ', inp[c_count])
      return inp[c_count]
    else:
      # print('c return: ', inp)
      return inp
  else:
    # print('d return: ', inp)
    return inp

def recursive_conv(filters, kernel_size, conv_strides=(1,1), padding='valid', pad_bool=False, pad_size=(0,0),
  pool_size=(2,2), pool_strides=(2,2), dilation_rate=(1,1), filters_up=None, kernel_size_up=None, strides_up=None, dropout=0, 
  layercombo='capb', weight_decay=0., block_name='convblock'):
    def f(input):
      x = input
      g = lambda input,c_count: input[c_count] if type(input) is list else input

      if type(layercombo) is tuple:
        startx = x
        end_x = []
        for i in range(len(layercombo)):
          x = recursive_conv(h(filters,i), h(kernel_size,i), h(conv_strides,i), h(padding,i), h(pad_bool,i), h(pad_size,i), h(pool_size,i),
            h(pool_strides,i), h(dilation_rate,i), h(filters_up,i), h(kernel_size_up,i), h(strides_up,i), h(dropout,i), 
            layercombo[i], weight_decay, block_name='{}_par{}'.format(block_name, i+1))(startx)
          end_x.append(x)
        x = concatenate(end_x, axis=-1)
      elif type(layercombo) is list:
        for i in range(len(layercombo)):
          x = recursive_conv(h(filters,i), h(kernel_size,i), h(conv_strides,i), h(padding,i), h(pad_bool,i), h(pad_size,i), h(pool_size,i),
            h(pool_strides,i), h(dilation_rate,i), h(filters_up,i), h(kernel_size_up,i), h(strides_up,i), h(dropout,i), 
            layercombo[i], weight_decay, block_name='{}_str{}'.format(block_name, i+1))(x)
      else:
        x = alex_conv(filters, kernel_size, conv_strides, padding, pad_bool, pad_size, pool_size, pool_strides, dilation_rate, filters_up, kernel_size_up, strides_up, dropout, 
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
  pool_size=(2,2), pool_strides=(2,2), dilation_rate=(1,1), filters_up=None, kernel_size_up=None, strides_up=None, dropout=0, 
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
          print("block: ", block_name, "filters_up:", f(filters_up,u_count), "conv_size_up:", f(kernel_size_up,u_count), "strides_up:", f(strides_up,u_count))
          x = Conv2DTranspose(f(filters_up,u_count), f(kernel_size_up,u_count), strides=f(strides_up,u_count), padding='same', kernel_initializer='he_normal', 
            name='{}_convT{}'.format(block_name, u_count+1))(x)
          u_count +=1
      return x
    return f

def alex_parallelconv(filters, numbranches, kernel_size, conv_strides=(1,1), padding='valid', pad_bool=False, pad_size=(0,0), 
  pool_size=(2,2), pool_strides=(2,2), dilation_rate=(1,1), dropout=0, layercombo='capb', weight_decay=0., block_name='alexblock'):
    def f(input):
      start_x = input
      shortcut_startx = start_x
      end_x = []
      f = lambda input,c_count: input[c_count] if type(input) is list else input

      for i in range(numbranches):
        currentcombo = f(layercombo, i)
        c_total = currentcombo.count("c") # 2D convolution
        a_total = currentcombo.count("a") # Activation (relu)
        p_total = currentcombo.count("p") # Pool
        b_total = currentcombo.count("b") # Batch norm
        d_total = currentcombo.count("d") # Dropout
        z_total = currentcombo.count("z") # zero padding
        s_total = currentcombo.count("s") # residual shortcut connection
        c_count=0
        a_count=0
        p_count=0
        b_count=0
        d_count=0
        z_count=0
        s_count=0
        x = start_x
        shortcut_startx = start_x

        for layer_char in currentcombo:
          if layer_char == "z":
            x = ZeroPadding2D(padding=f(f(pad_size,i),z_count), name="{}_parallel{}_Padding{}".format(block_name,i+1,z_count+1))(x)
            z_count +=1

          if layer_char == "c":
            x = Conv2D(f(f(filters,i),c_count), f(f(kernel_size,i),c_count), strides=f(f(conv_strides,i),c_count), padding=f(f(padding,i),c_count), dilation_rate=f(f(dilation_rate,i),c_count),
              kernel_initializer='he_normal', kernel_regularizer=l2(weight_decay), name='{}_parallel{}_conv{}'.format(block_name,i+1,c_count+1))(x)
            c_count +=1

          if layer_char == "p":
            x = MaxPooling2D(pool_size=f(f(pool_size,i), p_count), padding='same', strides=pool_strides, name='{}_parallel{}_pool{}'.format(block_name,i+1,p_count+1))(x)
            p_count +=1

          if layer_char == "b":
            x = BatchNormalization(name='{}_parallel{}_BatchNorm{}'.format(block_name, i+1, b_count+1))(x)
            b_count +=1

          if layer_char == "a":
            x = Activation('relu', name='{}_parallel{}_Activ{}'.format(block_name, i+1, a_count+1))(x)
            a_count +=1

          if layer_char == "d":
            x = Dropout(f(f(dropout,i),d_count), name='{}_parallel{}_Dropout{}'.format(block_name, i+1, d_count+1))(x)
            d_count +=1

          if layer_char == "s":
            x = res_shortcut(shortcut_startx, x, weight_decay, block_name='{}_parallel{}_Shortcut{}'.format(block_name, i+1, s_count+1))
            shortcut_startx = x
            s_count +=1
        end_x.append(x)

      x = add(end_x)  # add all branches
      return x
    return f

def parallel_conv(filters, kernel_size, conv_strides=(1,1), padding='valid', pad_bool=False, pad_size=(0,0), pool_size=(2,2), pool_strides=(2,2), 
  dilation_rate=(1,1), dropout=[0.5], layercombo='cacac', weight_decay=0., block_name='parallel_convblock'):
    def f(input):

      g = lambda input,c_count: input[0] if len(input)==1 else input[c_count]
      if type(input) is list:
        n = len(input)
        print("WENT THERE")
        print(input)
        x = input
        # factor = [int(x_i.shape[1]//x[0].shape[1]) for x_i in x]

        for i in range(n):
          # NOTE!!! A list PASSES by reference, and hence will point to the same location!
          x[i] = alex_conv(filters, kernel_size, conv_strides=conv_strides, padding=padding, pad_bool=pad_bool, pad_size=pad_size, 
            pool_size=pool_size, pool_strides=pool_strides, dilation_rate=dilation_rate, dropout=dropout, layercombo=layercombo, weight_decay=weight_decay,
            block_name='{}_alexconv{}'.format(block_name,i+1))(x[i]) # first don't factor the pools, need to save them for later sections
        return x
      else:
        x = input
        n = len(layercombo)
        y = []

        # actual start of CNN
        for i in range(n):
          y.append(alex_conv(g(filters,i), g(kernel_size,i), conv_strides=g(conv_strides,i), padding=g(padding,i), pad_bool=pad_bool, pad_size=g(pad_size,i), 
            pool_size=g(pool_size,i), pool_strides=g(pool_strides,i), dilation_rate=g(dilation_rate,i), dropout=g(dropout,i), layercombo=layercombo[i], weight_decay=weight_decay, 
            block_name='{}_alexconv{}'.format(block_name,i+1))(x))
        return y
    return f

# deprecated, use alex_conv with layercombo instead
def vgg_convblock(filters, kernel_size, convs=1, conv_strides=(1,1), padding='same', pad_bool=False, pool_bool=True, batchnorm_bool=False, pad_size=(0,0),
  pool_size=(2,2), pool_strides=(2,2), dilation_rate=(1,1), weight_decay=0., block_name='vgg_convblock'):
    def f(input):
      x = input
      for i in range(convs):
        if i < convs-1:
          x = alex_conv(filters, kernel_size, padding=padding, pad_bool=pad_bool, pool_bool=False, batchnorm_bool=False, pad_size=pad_size,
            dilation_rate=dilation_rate, weight_decay=weight_decay, block_name='{}_alexconv{}'.format(block_name, int(i+1)))(x)
        else:
          x = alex_conv(filters, kernel_size, padding=padding, pad_bool=pad_bool, pool_bool=pool_bool, batchnorm_bool=batchnorm_bool, pad_size=pad_size,
            pool_size=pool_size, pool_strides=pool_size, dilation_rate=dilation_rate, weight_decay=weight_decay, 
            block_name='{}_alexconv{}'.format(block_name, int(i+1)))(x)
      return x
    return f

def vgg_deconvblock(classes, scale, bridge_params=None, prev_params=None, next_params=None, upsample=True, target_shape=None, weight_decay=0., block_name='vgg_deconvblock'):
  # bridge_params are organized as [filter, conv_size, layercombo]
    def f(x, y):
        if bridge_params is not None:
          x = alex_conv(bridge_params[0], bridge_params[1], padding='same', pad_size=(0,0), dilation_rate=(1,1), filters_up=bridge_params[2], kernel_size_up=bridge_params[3], strides_up=bridge_params[4],
            layercombo=bridge_params[5], weight_decay=weight_decay, block_name='{}_bridgeconv'.format(block_name))(x)

        # x = Conv2D(classes, kernel_size=(1,1), activation='linear', padding='valid', kernel_initializer='he_normal', kernel_regularizer=l2(weight_decay),
        #   name='{}_1b1conv'.format(block_name))(x)

        if y is not None:
          if prev_params is not None:
            y = alex_conv(prev_params[0], prev_params[1], padding='same', pad_size=(0,0), dilation_rate=(1,1), filters_up=prev_params[2], kernel_size_up=prev_params[3], strides_up=prev_params[4],
              layercombo=prev_params[5], weight_decay=weight_decay, block_name='{}_prevconv'.format(block_name))(y)
          def scaling(xx, ss=1):
            return xx * ss
          scaled = Lambda(scaling, arguments={'ss': scale}, name='{}_scale'.format(block_name))(x)
          x = add([y, scaled])

        if next_params is not None:
          x = alex_conv(next_params[0], next_params[1], padding='same', pad_size=(0,0), dilation_rate=(1,1), filters_up=next_params[2], kernel_size_up=next_params[3], strides_up=next_params[4],
            layercombo=next_params[5], weight_decay=weight_decay, block_name='{}_nextconv'.format(block_name))(x)

        if upsample:
          upscore = BilinearUpSampling2D(target_shape=target_shape, name='{}_BilinearUpsample'.format(block_name))(x)
        else:
          upscore = x
        return upscore
    return f


def pool_concat(pool_size=(1,1), batchnorm_bool=False, block_name='poolconcat_block'):
    def f(input):
      n = len(input)
      x = input
      factor = [int(x_i.shape[1]//x[0].shape[1]) for x_i in x]

      for i in range(n):
        pool_size_i = [factor[i]*k for k in pool_size]
        if pool_size_i[0] > x[i].shape[1]:
          temp_padsize = int(np.ceil((pool_size_i[0]-int(x[i].shape[1]))/2))
          x[i] = ZeroPadding2D(padding=(temp_padsize, temp_padsize))(x[i])
        x[i] = MaxPooling2D(pool_size=pool_size_i, strides=pool_size_i, name='{}_pool{}'.format(block_name,i+1))(x[i])

        if batchnorm_bool:
          x[i] = BatchNormalization()(x[i])
        if i!=0:
          x[0] = concatenate([x[0],x[i]], name='{}_concat{}'.format(block_name,i))

      return x[0]
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


def vgg_fcblock(filters, full_layers, dropout_bool=False, dropout=0.5, weight_decay=0., block_name='vgg_fcblock'):
    def f(input):
      x = input

      for i in range(full_layers):
        x = Conv2D(filters[i], kernel_size=(1,1), activation='relu', padding='same', kernel_initializer='he_normal', kernel_regularizer=l2(weight_decay),
          name='{}_1b1conv{}'.format(block_name,i+1))(x)
        if dropout_bool:
          x = Dropout(dropout[i])(x)
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

def res_initialconv(filters, init_kernel=(7,7), init_strides=(2,2), weight_decay=0., block_name='initblock'):
    """ First basic convolution that gets everything started. 
    Format is Conv (/2) -> BN -> Actv -> Pool (/2)
    """
    def f(input):
      x = Conv2D(filters, init_kernel, strides=init_strides, padding='same',
        kernel_initializer='he_normal', kernel_regularizer=l2(weight_decay),
        name='{}_conv'.format(block_name))(input)
      x = BatchNormalization()(x)
      x = Activation('relu')(x)
      x = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding="same", name='{}_pool'.format(block_name))(x)
      return x
    return f

def res_basicconv(filters, convs=2, init_strides=(1,1), weight_decay=0., block_name='blockx'):
    """ Basic 3 X 3 convolution blocks for use in resnets with layers <= 34.
    Follows improved proposed scheme in hhttp://arxiv.org/pdf/1603.05027v2.pdf
    Format is BN -> Actv -> Conv (/2, if first of block), except for the very first conv of first block (just Conv).
    """
    def f(input):
      for i in range(convs):
        if i == 0:
          x = input

        if block_name =='megablock1_block1' and i==0:
          x = Conv2D(filters, (3,3), strides=init_strides, padding='same',
            kernel_initializer='he_normal', kernel_regularizer=l2(weight_decay),
            name='{}_conv{}'.format(block_name, int(i+1)))(x)   # linear activation for very first conv of first block
        else:
          x = BatchNormalization()(x)
          x = Activation('relu')(x)
          if i==0:
            x = Conv2D(filters, (3,3), strides=init_strides, padding='same', 
              kernel_initializer='he_normal', kernel_regularizer=l2(weight_decay),
              name='{}_conv{}'.format(block_name, int(i+1)))(x)
          else:
            x = Conv2D(filters, (3,3), strides=(1,1), padding='same',
              kernel_initializer='he_normal', kernel_regularizer=l2(weight_decay),
              name='{}_conv{}'.format(block_name, int(i+1)))(x)
      return res_shortcut(input, x, weight_decay)
    return f

def res_megaconv(filters, convs, reps, init_strides=(1,1), weight_decay=0., block_name='megablockx'):
    """ Mega convolution block that combines multiple res_basicconv blocks. Note that init_strides must be defined for the first block
        in case downsampling is necessary
    """
    def f(input):
      x = input
      for i in range(reps):
        if i == 0:
          x = res_basicconv(filters, convs, init_strides=init_strides, weight_decay = weight_decay, block_name='{}_block{}'.format(block_name, int(i+1)))(x)
        else:
          x = res_basicconv(filters, convs, init_strides=(1,1), weight_decay = weight_decay, block_name='{}_block{}'.format(block_name, int(i+1)))(x)
      return x
    return f

def res_1b1conv(filters, convs, init_kernelsize=(1,1), init_dilationrate=(1,1), weight_decay=0., block_name='block1b1'):
    """ 1x1xM convolution filter for resnet
    """
    def f(input):
      for i in range(convs):
        if i==0:
          x = BatchNormalization()(input)
          x = Activation('relu')(x)
          x = Conv2D(filters, kernel_size=init_kernelsize, padding='same', dilation_rate=init_dilationrate,
            kernel_initializer='he_normal', kernel_regularizer=l2(weight_decay), name='{}_1b1conv{}'.format(block_name,int(i+1)))(x)
        else:
          x = Activation('relu')(x)
          x = Conv2D(filters, kernel_size=(1,1), activation='relu', padding='same',
            kernel_initializer='he_normal', kernel_regularizer=l2(weight_decay), name='{}_1b1conv{}'.format(block_name,int(i+1)))(x)
      return x
    return f

def res_fc(classes, weight_decay=0., block_name='blockfc'):
    """ Fully connected layer for resnet
    """
    def f(input):
      block_shape = K.int_shape(input)
      pool = AveragePooling2D(pool_size=(block_shape[1], block_shape[2]), strides=(1,1), name='{}_pool'.format(block_name))(input)
      flatten = Flatten()(pool)
      dense = Dense(units=classes, kernel_initializer='he_normal', kernel_regularizer=l2(weight_decay), name='{}_dense'.format(block_name))(flatten)
      return dense
    return f

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
