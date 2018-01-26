import numpy as np
import keras.backend as K
from keras.layers import Input
import sys
sys.path.append("..") # Adds higher directory to python modules path.
from NeMO_blocks import (
    vgg_conv,
    vgg_fc,
    parallel_conv,
    res_shortcut,
    res_initialconv,
    res_megaconv,
    res_basicconv,
    res_1b1conv,
    res_fc,
    vgg_deconv,
    vgg_score
)

def test_vgg_conv():
    if K.image_data_format() == 'channels_first':
        x = Input(shape=(3, 224, 224))
        y1_shape = (None, 64, 112, 112)
        y2_shape = (None, 128, 56, 56)
    else:
        x = Input(shape=(224, 224, 3))
        y1_shape = (None, 112, 112, 64)
        y2_shape = (None, 56, 56, 128)

    block1 = vgg_conv(filters=64, convs=2, block_name='block1')
    y = block1(x)
    assert K.int_shape(y) == y1_shape

    block2 = vgg_conv(filters=128, convs=2, block_name='block2')
    y = block2(y)
    assert K.int_shape(y) == y2_shape


def test_vgg_fc():
    if K.image_data_format() == 'channels_first':
        x1 = K.variable(np.random.random((1, 512, 14, 14)))
        y1_shape = (1, 512, 8, 8)
    else:
        x1 = K.variable(np.random.random((1, 14, 14, 512)))
        y1_shape = (1, 8, 8, 512)

def test_parallel_conv():
    x = [Input(shape=(25,25,8)), Input(shape=(50,50,8)), Input(shape=(100,100,8))]
    y_shape = [(None,9,9,64), (None,9,9,64), (None,9,9,64)]

    y = parallel_conv(filters=64, kernel_size=(7,7), pad_size=(0,0), pool_size=(2,2), dilation_rate=(1,1))(x)
    for count,y_i in enumerate(y):
        assert K.int_shape(y_i) == y_shape[count]

def test_res_shortcut():
    x1 = Input(shape=(56,56,64))
    x2 = Input(shape=(28,28,128))
    y_shape = (None, 28, 28, 128)

    y = res_shortcut(x1,x2)
    assert K.int_shape(y) == y_shape

def test_res_initialconv():
    x = Input(shape=(224, 224, 3))
    y_shape = (None, 56, 56, 64)

    initial_block = res_initialconv(filters=64, init_strides=(2,2), block_name='initblock')
    y = initial_block(x)
    assert K.int_shape(y) == y_shape

def test_res_basicconv():
    x = Input(shape=(56,56,64))
    y1_shape = (None, 56, 56, 64)
    y2_shape = (None, 28, 28, 128)

    block1 = res_basicconv(filters=64, convs=2, init_strides=(1,1), block_name='megablock1_block1')
    block2 = res_basicconv(filters=128, convs=2, init_strides=(2,2), block_name='megablock2_block1')
    y1 = block1(x)
    y2 = block2(x)
    assert K.int_shape(y1) == y1_shape
    assert K.int_shape(y2) == y2_shape

def test_res_megaconv():
    x = Input(shape=(56,56,64))
    y1_shape = (None, 56, 56, 64)
    y2_shape = (None, 28, 28, 128)

    block1 = res_megaconv(filters=64, convs=2, reps=3, init_strides=(1,1), block_name='megablock1')
    block2 = res_megaconv(filters=128, convs=2, reps=4, init_strides=(2,2), block_name='megablock2')
    y1 = block1(x)
    y2 = block2(x)
    assert K.int_shape(y1) == y1_shape
    assert K.int_shape(y2) == y2_shape

def test_res_1b1conv():
    x = Input(shape=(14,14,512))
    y1_shape = (None, 14, 14, 20)
    y2_shape = (None, 14, 14, 20)

    block1 = res_1b1conv(filters=20, convs=1, init_kernelsize=(1,1), init_dilationrate=(1,1), block_name='block1b1')
    block2 = res_1b1conv(filters=20, convs=2, init_kernelsize=(7,7), init_dilationrate=(2,2), block_name='block1b1')
    y1 = block1(x)
    y2 = block2(x)
    assert K.int_shape(y1) == y1_shape
    assert K.int_shape(y2) == y2_shape

def test_res_fc():
    x = Input(shape=(7,7,512))
    classes = 10
    y_shape = (None,10)

    block1 = res_fc(classes=classes, block_name='blockfc')
    y = block1(x)
    assert K.int_shape(y) == y_shape

# def test_vgg_deconv():
#     if K.image_data_format() == 'channels_first':
#         x1 = K.variable(np.random.random((1, 512, 8, 8)))
#         y1_shape = (1, 21, 18, 18)
#         x2 = K.variable(np.random.random((1, 512, 27, 27)))
#         y2_shape = (1, 21, 38, 38)
#         x3 = K.variable(np.random.random((1, 256, 53, 53)))
#         y3_shape = (1, 21, 312, 312)
#     else:
#         x1 = K.variable(np.random.random((1, 8, 8, 512)))
#         y1_shape = (1, 18, 18, 21)
#         x2 = K.variable(np.random.random((1, 27, 27, 512)))
#         y2_shape = (1, 38, 38, 21)
#         x3 = K.variable(np.random.random((1, 53, 53, 256)))
#         y3_shape = (1, 312, 312, 21)

#     upscore1 = vgg_deconv(classes=21)(x1, None)
#     assert K.int_shape(upscore1) == y1_shape
#     assert not np.any(np.isnan(K.eval(upscore1)))

#     upscore2 = vgg_deconv(classes=21)(x2, upscore1)
#     assert K.int_shape(upscore2) == y2_shape
#     assert not np.any(np.isnan(K.eval(upscore2)))

#     upscore3 = vgg_deconv(classes=21, kernel_size=(16, 16),
#                           strides=(8, 8))(x3, upscore2)
#     assert K.int_shape(upscore3) == y3_shape
#     assert not np.any(np.isnan(K.eval(upscore3)))


# def test_vgg_score():
    # if K.image_data_format() == 'channels_first':
    #     x1 = K.variable(np.random.random((1, 3, 224, 224)))
    #     x2 = K.variable(np.random.random((1, 21, 312, 312)))
    #     y_shape = (1, 21, 224, 224)
    # else:
    #     x1 = K.variable(np.random.random((1, 224, 224, 3)))
    #     x2 = K.variable(np.random.random((1, 312, 312, 21)))
    #     y_shape = (1, 224, 224, 21)
    # score = vgg_score(crop_offset='centered')(x1, x2)
    # assert K.int_shape(score) == y_shape
