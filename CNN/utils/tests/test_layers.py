import pytest
import numpy as np
import keras.backend as K
import sys
sys.path.append("..") # Adds higher directory to python modules path.
from NeMO_layers import CroppingLike2D, BilinearUpSampling2D
from keras.utils.test_utils import keras_test


@keras_test
def test_bilinear_upsampling_2d():
    num_samples = 2
    stack_size = 2
    input_len_dim1 = 5
    input_len_dim2 = 5
    target_len_dim1 = 8
    target_len_dim2 = 8

    for data_format in ['channels_first', 'channels_last']:
        if data_format == 'channels_first':
            inputs = np.random.rand(num_samples, stack_size,
                                    input_len_dim1, input_len_dim2)
            target = np.random.rand(num_samples, stack_size,
                                    target_len_dim1, target_len_dim2)
            expected_output_shape = (2, 2, 8, 8)
        else:
            inputs = np.random.rand(num_samples,
                                    input_len_dim1, input_len_dim2,
                                    stack_size)
            target = np.random.rand(num_samples, target_len_dim1,
                                    target_len_dim2, stack_size)
            expected_output_shape = (2, 8, 8, 2)
        # shape test
        layer = BilinearUpSampling2D(target_shape=target.shape,
                                     data_format=data_format)
        output = layer(K.variable(inputs))
        assert K.int_shape(output) == expected_output_shape


@keras_test
def test_cropping_like_2d():
    num_samples = 2
    stack_size = 2
    input_len_dim1 = 9
    input_len_dim2 = 9
    target_len_dim1 = 5
    target_len_dim2 = 5
    offset = (2, 2)

    for data_format in ['channels_first', 'channels_last']:
        if data_format == 'channels_first':
            inputs = np.random.rand(num_samples, stack_size,
                                    input_len_dim1, input_len_dim2)
            target = np.random.rand(num_samples, stack_size,
                                    target_len_dim1, target_len_dim2)
            invalid_target = np.random.rand(num_samples, stack_size,
                                            input_len_dim1 + 1,
                                            input_len_dim2 + 1)
            expected_output = inputs[:,
                                     :,
                                     offset[0]: offset[0] + target_len_dim1,
                                     offset[1]: offset[1] + target_len_dim2]
            expected_output_shape = (2, 2, 5, 5)
        else:
            inputs = np.random.rand(num_samples,
                                    input_len_dim1, input_len_dim2,
                                    stack_size)
            target = np.random.rand(num_samples, target_len_dim1,
                                    target_len_dim2, stack_size)
            invalid_target = np.random.rand(num_samples, input_len_dim1 + 1,
                                            input_len_dim2 + 1, stack_size)
            expected_output = inputs[:,
                                     offset[0]: offset[0] + target_len_dim1,
                                     offset[1]: offset[1] + target_len_dim2,
                                     :]
            expected_output_shape = (2, 5, 5, 2)
        # basic test
        layer = CroppingLike2D(target_shape=target.shape,
                               offset=2, data_format=data_format)
        assert layer.offset == (2, 2)
        layer = CroppingLike2D(target_shape=target.shape,
                               offset=offset, data_format=data_format)
        # correctness test
        output = layer(K.variable(inputs))
        assert K.int_shape(output) == expected_output_shape
        np_output = K.eval(output)
        assert np.allclose(np_output, expected_output)
        # 'centered' test
        layer = CroppingLike2D(target_shape=target.shape,
                               offset='centered', data_format=data_format)
        output = layer(K.variable(inputs))
        np_output = K.eval(output)
        assert np.allclose(np_output, expected_output)

        # Test invalid use cases
        with pytest.raises(ValueError):
            layer = CroppingLike2D(target_shape=target.shape,
                                   offset=(5, 2),
                                   data_format=data_format)
            output = layer(K.variable(inputs))
        with pytest.raises(ValueError):
            layer = CroppingLike2D(target_shape=target.shape,
                                   offset=(2, 5),
                                   data_format=data_format)
            output = layer(K.variable(inputs))
        with pytest.raises(ValueError):
            layer = CroppingLike2D(target_shape=target.shape,
                                   offset=[3, 3, 3],
                                   data_format=data_format)
        with pytest.raises(ValueError):
            layer = CroppingLike2D(target_shape=invalid_target.shape,
                                   offset='centered', data_format=data_format)
            output = layer(K.variable(inputs))
