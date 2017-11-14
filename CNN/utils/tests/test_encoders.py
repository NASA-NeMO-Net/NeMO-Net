import numpy as np
import keras.backend as K
from keras.layers import Input
import sys
sys.path.append("..") # Adds higher directory to python modules path.
from NeMO_encoders import (
    VGG16,
    VGG19)

from keras.utils.test_utils import keras_test


@keras_test
def test_vgg16():
    for data_format in ['channels_first', 'channels_last']:
        K.set_image_data_format(data_format)
        if K.image_data_format() == 'channels_first':
            x = Input(shape=(3, 500, 500))
            pool1_shape = (None, 64, 250, 250)
            pool2_shape = (None, 128, 125, 125)
            pool3_shape = (None, 256, 63, 63)
            pool4_shape = (None, 512, 32, 32)
            drop7_shape = (None, 4096, 16, 16)
            conv1_weight = -0.35009676
        else:
            x = Input(shape=(500, 500, 3))
            pool1_shape = (None, 250, 250, 64)
            pool2_shape = (None, 125, 125, 128)
            pool3_shape = (None, 63, 63, 256)
            pool4_shape = (None, 32, 32, 512)
            drop7_shape = (None, 16, 16, 4096)
            conv1_weight = 0.429471

        encoder = VGG16(x, weights='imagenet', trainable=False)
        feat_pyramid = encoder.outputs

        assert len(feat_pyramid) == 5

        assert K.int_shape(feat_pyramid[0]) == drop7_shape
        assert K.int_shape(feat_pyramid[1]) == pool4_shape
        assert K.int_shape(feat_pyramid[2]) == pool3_shape
        assert K.int_shape(feat_pyramid[3]) == pool2_shape
        assert K.int_shape(feat_pyramid[4]) == pool1_shape

        for layer in encoder.layers:
            if layer.name == 'block1_conv1':
                assert layer.trainable is False
                weights = K.eval(layer.weights[0])
                assert np.allclose(weights[0, 0, 0, 0], conv1_weight)

        encoder_from_scratch = VGG16(x, weights=None, trainable=True)
        for layer in encoder_from_scratch.layers:
            if layer.name == 'block1_conv1':
                assert layer.trainable is True
                weights = K.eval(layer.weights[0])
                assert not np.allclose(weights[0, 0, 0, 0], conv1_weight)


@keras_test
def test_vgg19():
    for data_format in ['channels_first', 'channels_last']:
        K.set_image_data_format(data_format)
        if K.image_data_format() == 'channels_first':
            x = Input(shape=(3, 500, 500))
            pool1_shape = (None, 64, 250, 250)
            pool2_shape = (None, 128, 125, 125)
            pool3_shape = (None, 256, 63, 63)
            pool4_shape = (None, 512, 32, 32)
            drop7_shape = (None, 4096, 16, 16)
            conv1_weight = -0.35009676
        else:
            x = Input(shape=(500, 500, 3))
            pool1_shape = (None, 250, 250, 64)
            pool2_shape = (None, 125, 125, 128)
            pool3_shape = (None, 63, 63, 256)
            pool4_shape = (None, 32, 32, 512)
            drop7_shape = (None, 16, 16, 4096)
            conv1_weight = 0.429471

        encoder = VGG19(x, weights='imagenet', trainable=False)
        feat_pyramid = encoder.outputs

        assert len(feat_pyramid) == 5

        assert K.int_shape(feat_pyramid[0]) == drop7_shape
        assert K.int_shape(feat_pyramid[1]) == pool4_shape
        assert K.int_shape(feat_pyramid[2]) == pool3_shape
        assert K.int_shape(feat_pyramid[3]) == pool2_shape
        assert K.int_shape(feat_pyramid[4]) == pool1_shape

        for layer in encoder.layers:
            if layer.name == 'block1_conv1':
                assert layer.trainable is False
                weights = K.eval(layer.weights[0])
                assert np.allclose(weights[0, 0, 0, 0], conv1_weight)

        encoder_from_scratch = VGG19(x, weights=None, trainable=True)
        for layer in encoder_from_scratch.layers:
            if layer.name == 'block1_conv1':
                assert layer.trainable is True
                weights = K.eval(layer.weights[0])
                assert not np.allclose(weights[0, 0, 0, 0], conv1_weight)
