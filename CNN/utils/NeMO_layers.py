import keras.backend as K
import NeMO_backend as K1
from keras.utils import conv_utils
from keras.engine.topology import Layer
from keras.engine import InputSpec
import tensorflow as tf

from hyperopt import fmin, tpe, hp, STATUS_OK, Trials


class BilinearUpSampling2D(Layer):
    """Upsampling2D with bilinear interpolation."""

    def __init__(self, target_shape=None, data_format=None, method='bilinear', **kwargs):
        if data_format is None:
            data_format = K.image_data_format()
        assert data_format in {
            'channels_last', 'channels_first'}
        self.data_format = data_format
        self.input_spec = [InputSpec(ndim=4)]
        self.target_shape = target_shape
        self.method = method
        if self.data_format == 'channels_first':
            self.target_size = (target_shape[2], target_shape[3])
        elif self.data_format == 'channels_last':
            self.target_size = (target_shape[1], target_shape[2])
        super(BilinearUpSampling2D, self).__init__(**kwargs)

    def compute_output_shape(self, input_shape):
        if self.data_format == 'channels_last':
            return (input_shape[0], self.target_size[0],
                    self.target_size[1], input_shape[3])
        else:
            return (input_shape[0], input_shape[1],
                    self.target_size[0], self.target_size[1])

    def call(self, inputs):
        return K1.resize_images(inputs, size=self.target_size,
                                method=self.method)

    def get_config(self):
        config = {'target_shape': self.target_shape,
                'data_format': self.data_format}
        base_config = super(BilinearUpSampling2D, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class CroppingLike2D(Layer):
    def __init__(self, target_shape, offset=None, data_format=None,
                 **kwargs):
        """Crop to target.

        If only one `offset` is set, then all dimensions are offset by this amount.

        """
        super(CroppingLike2D, self).__init__(**kwargs)
        self.data_format = conv_utils.normalize_data_format(data_format)
        self.target_shape = target_shape
        if offset is None or offset == 'centered':
            self.offset = 'centered'
        elif isinstance(offset, int):
            self.offset = (offset, offset)
        elif hasattr(offset, '__len__'):
            if len(offset) != 2:
                raise ValueError('`offset` should have two elements. '
                                 'Found: ' + str(offset))
            self.offset = offset
        self.input_spec = InputSpec(ndim=4)

    def compute_output_shape(self, input_shape):
        if self.data_format == 'channels_first':
            return (input_shape[0],
                    input_shape[1],
                    self.target_shape[2],
                    self.target_shape[3])
        else:
            return (input_shape[0],
                    self.target_shape[1],
                    self.target_shape[2],
                    input_shape[3])

    def call(self, inputs):
        input_shape = K.int_shape(inputs)
        if self.data_format == 'channels_first':
            input_height = input_shape[2]
            input_width = input_shape[3]
            target_height = self.target_shape[2]
            target_width = self.target_shape[3]
            if target_height > input_height or target_width > input_width:
                raise ValueError('The Tensor to be cropped need to be smaller'
                                 'or equal to the target Tensor.')

            if self.offset == 'centered':
                self.offset = [int((input_height - target_height) / 2),
                               int((input_width - target_width) / 2)]

            if self.offset[0] + target_height > input_height:
                raise ValueError('Height index out of range: '
                                 + str(self.offset[0] + target_height))
            if self.offset[1] + target_width > input_width:
                raise ValueError('Width index out of range:'
                                 + str(self.offset[1] + target_width))

            return inputs[:,
                          :,
                          self.offset[0]:self.offset[0] + target_height,
                          self.offset[1]:self.offset[1] + target_width]
        elif self.data_format == 'channels_last':
            input_height = input_shape[1]
            input_width = input_shape[2]
            target_height = self.target_shape[1]
            target_width = self.target_shape[2]
            if target_height > input_height or target_width > input_width:
                raise ValueError('The Tensor to be cropped need to be smaller'
                                 'or equal to the target Tensor.')

            if self.offset == 'centered':
                self.offset = [int((input_height - target_height) / 2),
                               int((input_width - target_width) / 2)]

            if self.offset[0] + target_height > input_height:
                raise ValueError('Height index out of range: '
                                 + str(self.offset[0] + target_height))
            if self.offset[1] + target_width > input_width:
                raise ValueError('Width index out of range:'
                                 + str(self.offset[1] + target_width))
            output = inputs[:,
                            self.offset[0]:self.offset[0] + target_height,
                            self.offset[1]:self.offset[1] + target_width,
                            :]
            return output

    def get_config(self):
        config = {'target_shape': self.target_shape,
                  'offset': self.offset,
                  'data_format': self.data_format}
        base_config = super(CroppingLike2D, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
    
def gram_matrix(x): # Gram matrix of size [N_filter, rows x cols)], which contains the flattened features dotted with their transposes
    input_shape = K.int_shape(x)
    features = K.permute_dimensions(x, (0,3,1,2))
    features = K.reshape(features, (K.shape(x)[0], input_shape[3], input_shape[1]*input_shape[2]))
    gram = K.batch_dot(features, K.permute_dimensions(features, (0,2,1)))
    return gram

def style_loss(style, combination):
    S = gram_matrix(style)
    C = gram_matrix(combination)
    input_shape = K.int_shape(style)
    nrows = input_shape[1]
    ncols = input_shape[2]
    nchannels = input_shape[3]
    size = nrows*ncols
    return K.sum(K.batch_flatten(K.square(S - C)), axis=-1, keepdims=True) / (4.0 * (nchannels ** 2) * (size ** 2))

class Gram_Loss(Layer):
    ''' Implements gram matrix feature loss'''
    def __init__(self, weight, **kwargs):
        self.weight = weight
        super(Gram_Loss, self).__init__(**kwargs)
        
    def build(self, input_shape):
        self.trainable_weights = []
    
    def call(self, inputs):
        if type(inputs) is not list:
            raise Exception('Gram Loss must be between a two tensors: the original style vs the derived product')
        style_input = inputs[0]
        prod_input = inputs[1]
        assert(K.int_shape(style_input) == K.int_shape(prod_input)) # same input shapes
        input_shape = K.int_shape(style_input)
        
#         temp = K.permute_dimensions(style_input, (0,3,1,2))
#         temp = K.reshape(temp, (K.shape(style_input)[0], input_shape[3], input_shape[1]*input_shape[2]))
#         gram = K.batch_dot(temp, K.permute_dimensions(temp, (0,2,1)))
          
#         temp2 = K.permute_dimensions(prod_input, (0,3,1,2))
#         temp2 = K.reshape(temp2, (K.shape(prod_input)[0], input_shape[3], input_shape[1]*input_shape[2]))
#         gram2 = K.batch_dot(temp2, K.permute_dimensions(temp2, (0,2,1)))
        
#         combine = K.sum(K.batch_flatten(K.square(gram - gram2)),axis=-1)
        loss = self.weight*style_loss(style_input, prod_input)
        
#         loss = K.zeros(shape=(None,1))
#         if input_shape[0] is not None: # go over every individual batch
#             loss = K.zeros(shape=(input_shape[0],1)) # num_batch x 1
#             for i in range(input_shape[0]):
#                 K.sum(loss[i], style_loss(style_input[i], prod_input[i]))
        return loss
        
    def compute_output_shape(self, input_shape):
        return (input_shape[0][0],1)
    
    def get_config(self):
        config = {'weight': self.weight}
        base_config = super(Gram_Loss, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

class Var_Loss(Layer):
    def __init__(self, weight, **kwargs):
        self.weight = weight
        super(Var_Loss, self).__init__(**kwargs)
        
    def build(self, input_shape):
        self.trainable_weights = []
    
    def call(self, inputs):
        input_shape = K.int_shape(inputs)
        
        a = K.square(inputs[:, :input_shape[1]-1, :input_shape[2]-1, :] - inputs[:, 1:, :input_shape[2]-1, :])
        b = K.square(inputs[:, :input_shape[1]-1, :input_shape[2]-1, :] - inputs[:, :input_shape[1]-1, 1:, :])
        
        loss = self.weight*K.sum(K.batch_flatten(K.pow(a+b, 1.25)), axis=-1, keepdims=True)
        return loss
    
    def compute_output_shape(self, input_shape):
        return(input_shape[0],1)
    
    def get_config(self):
        config = {'weight': self.weight}
        base_config = super(Var_Loss, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
    
class Content_Loss(Layer):
    def __init__(self, weight, **kwargs):
        self.weight = weight
        super(Content_Loss, self).__init__(**kwargs)
        
    def build(self, input_shape):
        self.trainable_weights = []
    
    def call(self, inputs):
        if type(inputs) is not list:
            raise Exception('Content Loss must be between a two tensors: the original style vs the derived product')
        content_input = inputs[0]
        prod_input = inputs[1]
        assert(K.int_shape(content_input) == K.int_shape(prod_input)) # same input shapes
        input_shape = K.int_shape(content_input)
        
        loss = self.weight*(K.mean(K.batch_flatten(K.square(content_input - prod_input)), axis=-1, keepdims=True))
        return loss
    
    def compute_output_shape(self, input_shape):
        return(input_shape[0][0],1)
    
    def get_config(self):
        config = {'weight': self.weight}
        base_config = super(Content_Loss, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
    
def tfsplit(x):    
    x_shape = K.int_shape(x)
    if (x_shape is not None) and (x_shape[0] is not None):
        len_start = int_shape(start)[0] if K.is_tensor(start) else len(start)
        len_size = int_shape(size)[0] if K.is_tensor(size) else len(size)
        if not (len(K.int_shape(x)) == len_start == len_size):
            raise ValueError('The dimension and the size of indices should match.')
    split1, split2 = tf.split(x, num_or_size_splits=2, axis=0)
    return split1
    
class Batch_Split(Layer):
    '''Splits batch into two, but returns only first half'''
    def __init__(self, **kwargs):
        super(Batch_Split, self).__init__(**kwargs)
    
    def build(self, input_shape):
        self.trainable_weights = []
    
    def call(self, inputs):
        input_shape = K.int_shape(inputs)
        total_batch_size = input_shape[0]
        # assume size of tensor is 4 (num_batch x cols x rows x depth)
        return tfsplit(inputs)
        
    def compute_output_shape(self, input_shape):
        return input_shape
    
    
def reverse_gradient(X, hp_lambda):
    '''Flips the sign of the incoming gradient during training.'''
    try:
        reverse_gradient.num_calls += 1
    except AttributeError:
        reverse_gradient.num_calls = 1

    grad_name = "GradientReversal%d" % reverse_gradient.num_calls
    @tf.RegisterGradient(grad_name)
    def _flip_gradients(op, grad):
        return [tf.negative(grad) * hp_lambda]

    g = K.get_session().graph
    with g.gradient_override_map({'Identity': grad_name}):
        y = tf.identity(X)

    return y

class GradientReversal(Layer):
    '''Flip the sign of gradient during training.'''
    def __init__(self, hp_lambda, **kwargs):
        super(GradientReversal, self).__init__(**kwargs)
        self.supports_masking = False
        self.hp_lambda = hp_lambda

    def build(self, input_shape):
        self.trainable_weights = []

    def call(self, x, mask=None):
        return reverse_gradient(x, self.hp_lambda)

    def get_output_shape_for(self, input_shape):
        return input_shape
    
    def compute_output_shape(self, input_shape):
        return input_shape

    def get_config(self):
        config = {'hp_lambda': self.hp_lambda}
        base_config = super(GradientReversal, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
