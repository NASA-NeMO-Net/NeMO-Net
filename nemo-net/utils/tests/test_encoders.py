import numpy as np
import keras.backend as K
from keras.layers import Input
import sys
import numpy as np
sys.path.append("..") # Adds higher directory to python modules path.
from NeMO_encoders import (
    Res34_Encoder,
    Alex_Parallel_Hyperopt_Encoder,
    VGG16,
    VGG19)

from keras.utils.test_utils import keras_test

@keras_test
def test_Alex_Parallel():
    # x = [Input(shape=(25,25,8)), Input(shape=(50,50,8)), Input(shape=(100,100,8))]
    x = Input(shape=(100,100,8))
    crop_shapes = [(25,25),(50,50),(100,100)]
    classes = 24

    conv_params= {"filters": [64,128,256],
        "conv_size": [(7,7),(3,3),(3,3)],
        "dilation_rate": [(1,1),(1,1),(1,1)],
        "pool_size": [(2,2),(2,2),(1,1)],
        "pad_size": [(0,0),(0,0),(0,0)],
        "batchnorm_bool": [True,True,False],
        "full_filters": [2048],
        "dropout": [0.5]}

    p1_shape = [(None,1,1,256), (None,3,3,128), (None,9,9,64)] # smallest parallel branch
    p2_shape = [(None,3,3,256), (None,7,7,128), (None,19,19,64)] # middle parallel branch
    p3_shape = [(None,5,5,256), (None,15,15,128), (None,38,38,64)] # largest parallel branch
    pc_shape = (None,1,1,768)
    fc_shape = (None,2048)

    encoder = Alex_Parallel_Hyperopt_Encoder(x, crop_shapes, classes, weights=None, weight_decay=0., trainable=True, conv_layers=3, full_layers=1, conv_params=conv_params)
    feat_pyramid = encoder.outputs

    offset = 2
    input_len = len(crop_shapes)
    for i in range(input_len):
        assert K.int_shape(feat_pyramid[i*input_len+offset]) == p1_shape[i]
        assert K.int_shape(feat_pyramid[i*input_len+1+offset]) == p2_shape[i]
        assert K.int_shape(feat_pyramid[i*input_len+2+offset]) == p3_shape[i]
    assert K.int_shape(feat_pyramid[1]) == pc_shape
    assert K.int_shape(feat_pyramid[0]) == fc_shape

@keras_test
def test_Res34():
    x = Input(shape=(224, 224, 3))
    classes =4
    initconv_shape = (None, 56, 56, 64)
    megaconv1_shape = (None, 56, 56, 64)
    megaconv2_shape = (None, 28, 28, 128)
    megaconv3_shape = (None, 14, 14, 256)
    megaconv4_shape = (None, 7, 7, 512)
    conv1b1_shape = (None, 7, 7, 512)
    fc_shape = (None,classes)

    encoder = Res34_Encoder(x, classes, weights=None, trainable=True, fcflag=False)
    feat_pyramid = encoder.outputs

    assert len(feat_pyramid) == 6
    assert K.int_shape(feat_pyramid[0]) == conv1b1_shape
    assert K.int_shape(feat_pyramid[1]) == megaconv4_shape
    assert K.int_shape(feat_pyramid[2]) == megaconv3_shape
    assert K.int_shape(feat_pyramid[3]) == megaconv2_shape
    assert K.int_shape(feat_pyramid[4]) == megaconv1_shape
    assert K.int_shape(feat_pyramid[5]) == initconv_shape

    encoder2 = Res34_Encoder(x, classes, weights=None, trainable=True, fcflag=True)
    feat_pyramid2 = encoder2.outputs
    assert K.int_shape(feat_pyramid2[0]) == fc_shape

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
