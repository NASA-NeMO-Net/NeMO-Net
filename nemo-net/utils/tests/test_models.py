"""Test FCN."""
import numpy as np
import sys
sys.path.append("..") # Adds higher directory to python modules path.
from NeMO_models import FCN, ResNet34, Alex_Hyperopt_ParallelNet
from keras import backend as K


def is_same_shape(shape, expected_shape, data_format=None):
    """Test helper."""
    if data_format is None:
        data_format = K.image_data_format()
    if data_format == 'channels_first':
        expected_shape = (expected_shape[0],
                          expected_shape[3],
                          expected_shape[1],
                          expected_shape[2])
    return shape == expected_shape

def test_Alex_Hyperopt_ParallelNet():
    input_shape = (150,150,8)
    crop_shapes = [(25,25), (50,50), (100,100)]
    alex_parallelNet = Alex_Hyperopt_ParallelNet(input_shape=input_shape, crop_shapes=crop_shapes, classes=24, conv_layers=3, full_layers=1)

    for l in alex_parallelNet.layers:
        # print("LAYER NAME: ", l.name)
        # print("LAYER SHAPE: ", l.output_shape)
        if l.name == 'parallel_block1_alexconv1_pool':
            assert is_same_shape(l.output_shape, (None,9,9,64))
        if l.name == 'parallel_block1_alexconv2_pool':
            assert is_same_shape(l.output_shape, (None,19,19,64))
        if l.name == 'parallel_block1_alexconv3_pool':
            assert is_same_shape(l.output_shape, (None,38,38,64))
        if l.name == 'parallel_block2_alexconv1_pool':
            assert is_same_shape(l.output_shape, (None,3,3,128))
        if l.name == 'parallel_block2_alexconv2_pool':
            assert is_same_shape(l.output_shape, (None,7,7,128))
        if l.name == 'parallel_block2_alexconv3_pool':
            assert is_same_shape(l.output_shape, (None,15,15,128))
        if l.name == 'parallel_block3_alexconv1_pool':
            assert is_same_shape(l.output_shape, (None,1,1,256))
        if l.name == 'parallel_block3_alexconv2_pool':
            assert is_same_shape(l.output_shape, (None,3,3,256))
        if l.name == 'parallel_block3_alexconv3_pool':
            assert is_same_shape(l.output_shape, (None,5,5,256))
        if l.name == 'poolconcat_block_concat2':
            assert is_same_shape(l.output_shape, (None,1,1,768))
        if l.name == 'alexfc1_dense':
            assert is_same_shape(l.output_shape, (None,4096))
    assert is_same_shape(alex_parallelNet.output_shape, (None,24))

def test_Resnet34():
    input_shape = (224, 224, 3)
    resnet34 = ResNet34(input_shape=input_shape, classes=4)

    for l in resnet34.layers:
        if l.name == 'initblock_conv':
            test_shape = (None, 112, 112, 64)
            assert is_same_shape(l.output_shape, test_shape)
        elif l.name == 'initblock_pool':
            test_shape = (None, 56, 56, 64)
            assert is_same_shape(l.output_shape, test_shape)
        elif 'megablock1' in l.name:
            test_shape = (None, 56, 56, 64)
            assert is_same_shape(l.output_shape, test_shape)
        elif 'megablock2' in l.name:
            test_shape = (None, 28, 28, 128)
            assert is_same_shape(l.output_shape, test_shape)
        elif 'megablock3' in l.name:
            test_shape = (None, 14, 14, 256)
            assert is_same_shape(l.output_shape, test_shape)
        elif 'megablock4' in l.name:
            test_shape = (None, 7, 7, 512)
            assert is_same_shape(l.output_shape, test_shape)
        elif l.name == 'blockfc_pool':
            test_shape = (None, 1, 1, 512)
            assert is_same_shape(l.output_shape, test_shape)
        elif l.name == 'blockfc_dense':
            test_shape = (None, 4)
            assert is_same_shape(l.output_shape, test_shape)
    assert is_same_shape(resnet34.output_shape, (None, 4))


def test_fcn_vgg16_shape():
    """Test output shape."""
    if K.image_data_format() == 'channels_first':
        input_shape = (3, 500, 500)
    else:
        input_shape = (500, 500, 3)
    fcn_vgg16 = FCN(input_shape=input_shape, classes=21)

    layers = [l.name for l in fcn_vgg16.layers]
    assert 'upscore_feat1' in layers
    assert 'upscore_feat2' in layers
    assert 'upscore_feat3' in layers

    for l in fcn_vgg16.layers:
        if l.name == 'block1_pool':
            test_shape = (None, 250, 250, 64)
            assert is_same_shape(l.output_shape, test_shape)
        elif l.name == 'block2_pool':
            test_shape = (None, 125, 125, 128)
            assert is_same_shape(l.output_shape, test_shape)
        elif l.name == 'block3_pool':
            test_shape = (None, 63, 63, 256)
            assert is_same_shape(l.output_shape, test_shape)
        elif l.name == 'block4_pool':
            test_shape = (None, 32, 32, 512)
            assert is_same_shape(l.output_shape, test_shape)
        elif l.name == 'block5_pool':
            test_shape = (None, 16, 16, 512)
            assert is_same_shape(l.output_shape, test_shape)
        elif l.name == 'dropout_2':
            test_shape = (None, 16, 16, 4096)
            assert is_same_shape(l.output_shape, test_shape)
        elif l.name == 'upscore_feat1':
            test_shape = (None, 32, 32, 21)
            assert is_same_shape(l.output_shape, test_shape)
        elif l.name == 'upscore_feat2':
            test_shape = (None, 63, 63, 21)
            assert is_same_shape(l.output_shape, test_shape)
        elif l.name == 'upscore_feat3':
            test_shape = (None, 500, 500, 21)
            assert is_same_shape(l.output_shape, test_shape)
        elif l.name == 'score':
            test_shape = (None, 500, 500, 21)
            assert is_same_shape(l.output_shape, test_shape)
    assert is_same_shape(fcn_vgg16.output_shape, (None, 500, 500, 21))

    input_shape = (1366, 768, 3)
    fcn_vgg16 = FCN(input_shape=input_shape, classes=21)
    assert is_same_shape(fcn_vgg16.output_shape, (None, 1366, 768, 21))


def test_fcn_vgg16_correctness():
    """Test output not NaN."""
    if K.image_data_format() == 'channels_first':
        input_shape = (3, 500, 500)
        x = np.random.rand(1, 3, 500, 500)
        y = np.random.randint(21, size=(1, 500, 500))
        y = np.eye(21)[y]
        y = np.transpose(y, (0, 3, 1, 2))
    else:
        input_shape = (500, 500, 3)
        x = np.random.rand(1, 500, 500, 3)
        y = np.random.randint(21, size=(1, 500, 500))
        y = np.eye(21)[y]
    fcn_vgg16 = FCN(classes=21, input_shape=input_shape)
    fcn_vgg16.compile(optimizer='rmsprop',
                      loss='categorical_crossentropy',
                      metrics=['accuracy'])
    fcn_vgg16.fit(x, y, batch_size=1, epochs=1)
    loss = fcn_vgg16.evaluate(x, y, batch_size=1)
    assert not np.any(np.isinf(loss))
    assert not np.any(np.isnan(loss))
    y_pred = fcn_vgg16.predict(x, batch_size=1)
    assert not np.any(np.isinf(y_pred))
    assert not np.any(np.isnan(y_pred))
