import keras.backend as K
import numpy as np
from keras.layers import (
    Dropout,
    Lambda,
    Activation,
    Dense,
    Flatten
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

def alex_conv(filters, kernel_size, conv_strides=(1,1), pad_bool=False, pool_bool=False, batchnorm_bool = False, pad_size=(0,0), 
  pool_size=(2,2), pool_strides=(2,2), dilation_rate=(1,1), weight_decay=0., block_name='alexblock'):
    def f(input):
      x = input
      if pad_bool:
        x = ZeroPadding2D(padding=pad_size)(x)
        if kernel_size[0] > x.shape[1]:
          temp_padsize = int(np.ceil((kernel_size[0]-int(x.shape[1]))/2))
          x = ZeroPadding2D(padding=(temp_padsize,temp_padsize))(x)

      x = Conv2D(filters, kernel_size, strides=conv_strides, dilation_rate=dilation_rate, activation='relu',
        kernel_initializer='he_normal', kernel_regularizer=l2(weight_decay),
        name='{}_conv'.format(block_name))(x)

      if pool_bool:
        if pool_size[0] > x.shape[1]:
          temp_padsize = int(np.ceil((pool_size[0]-int(x.shape[1]))/2))
          x = ZeroPadding2D(padding=(temp_padsize,temp_padsize))(x)
        x = MaxPooling2D(pool_size=pool_size, strides=pool_strides, name='{}_pool'.format(block_name))(x)
      if batchnorm_bool:
        x = BatchNormalization()(x)
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

def res_shortcut(input, residual, weight_decay=0):
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
        padding="valid", kernel_initializer="he_normal", kernel_regularizer=l2(weight_decay))(input)

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
