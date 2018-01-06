from __future__ import (
    absolute_import,
    unicode_literals
)
import keras
import keras.backend as K

from keras.models import Model
from keras.utils.data_utils import get_file
from keras.utils import layer_utils

from NeMO_blocks import (
    alex_conv,
    alex_fc,
    res_initialconv,
    res_basicconv,
    res_megaconv,
    res_1b1conv,
    res_fc,
    vgg_conv,
    vgg_fc
)
from NeMO_backend import load_weights


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

    def __init__(self, inputs, blocks, weights=None,
                 trainable=True, name='encoder'):
        inverse_pyramid = []

        # all blocks
        for i, block in enumerate(blocks):
            if i == 0:
                x = block(inputs)
                inverse_pyramid.append(x)
            else:
                x = block(x)
                inverse_pyramid.append(x)

        outputs = list(reversed(inverse_pyramid))

        super(Res_Encoder, self).__init__(
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

class Alex_Encoder(Res_Encoder):
    def __init__(self, inputs, classes, weight_decay=0., weights=None, trainable=True):
        filters = [96, 256, 384, 384, 256]
        conv_size = [(7,7),(5,5),(3,3),(3,3),(3,3)]
        pool_size = [(2,2),(2,2),(1,1),(1,1),(2,2)]
        pool_stride = [(2,2),(2,2),(1,1),(1,1),(2,2)]
        pool_bool = [True, True, False, False, True]
        pad_bool = [False, False, True, True, True]
        batchnorm_bool = [True, True, False, False, False]

        full_filters = [4096, 4096]
        drop_bool = [True, True]
        drop_val = [0.5, 0.5]
        
        blocks = []
        for i in range(len(filters)):
            block_name = 'alexblock{}'.format(i + 1)
            block = alex_conv(filters[i], conv_size[i], pad_bool=pad_bool[i], pool_bool=pool_bool[i], batchnorm_bool=batchnorm_bool[i], pool_size=pool_size[i], pool_strides=pool_stride[i], weight_decay=weight_decay, block_name=block_name)
            blocks.append(block)

        for i in range(len(full_filters)):
            block_name='alexfc{}'.format(i + 1)
            if i==0:
                block = alex_fc(full_filters[i], flatten_bool=True, dropout_bool=drop_bool[i], dropout=drop_val[i], weight_decay=weight_decay, block_name=block_name)
            else:
                block = alex_fc(full_filters[i], flatten_bool=False, dropout_bool=drop_bool[i], dropout=drop_val[i], weight_decay=weight_decay, block_name=block_name)
            blocks.append(block)

        super(Alex_Encoder, self).__init__(inputs=inputs, blocks=blocks, weights=weights, trainable = trainable)


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
        fc_block = vgg_fc(filters=4096, weight_decay=weight_decay)
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
                                    filters=[64, 128, 256, 512, 512],
                                    convs=[2, 2, 3, 3, 3],
                                    weight_decay=weight_decay,
                                    weights=weights,
                                    trainable=trainable)


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
