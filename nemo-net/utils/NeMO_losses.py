import keras
import keras.backend as K
from keras.applications.vgg19 import preprocess_input, VGG19
from keras.models import Model
from keras.losses import mean_squared_error, mean_absolute_error
import numpy as np
import tensorflow as tf

from hyperopt import fmin, tpe, hp, STATUS_OK, Trials

def vgg19_content_loss(y_true, y_pred):
    # can't use imagenet weights, since we have (256 x 256 x 4) input
    
    input_shape = (256,256,3)
    vgg = VGG19(weights='imagenet', include_top=False, input_shape=input_shape)
    vgg_layer = vgg.layers[20].output
    
    mdl = Model(vgg.input, vgg_layer)
    mdl.trainable = False
    
    # y_true and y_pred has shape (?, 256, 256, 4)
    
#     y_pred = np.expand_dims(y_pred, axis=0)
#     y_true = np.expand_dims(y_true, axis=0)
#     sr = preprocess_input(y_pred)
#     hr = preprocess_input(y_true)
    sr_features = mdl(y_pred[:,:,:,:3])
    hr_features = mdl(y_true[:,:,:,:3])
    return mean_squared_error(hr_features, sr_features)

def model_content_loss(model, layername, weights=None):
    def loss(y_true, y_pred):
        output_layer = model.get_layer(layername).output
        mdl = Model(model.input, output_layer)
        mdl.trainable = False
        
        sr_features = mdl(y_pred)
        hr_features = mdl(y_true)
        
        if weights is not None:
            return weights[0]*K.mean(mean_absolute_error(y_true,y_pred), axis=(1,2)) + weights[1]*K.mean(mean_absolute_error(hr_features, sr_features), axis=(1,2))
        else:
            return mean_absolute_error(hr_features, sr_features)
    return loss

def mean_categorical_crossentropy(y_true, y_pred):
    if K.image_data_format() == 'channels_last':
        loss = K.mean(keras.losses.categorical_crossentropy(y_true, y_pred), axis=[1, 2])
    elif K.image_data_format() == 'channels_first':
        loss = K.mean(keras.losses.categorical_crossentropy(y_true, y_pred), axis=[2, 3])
    return loss

def flatten_categorical_crossentropy(classes):
    def f(y_true, y_pred):
        y_true = K.reshape(y_true, (-1, classes))
        y_pred = K.reshape(y_pred, (-1, classes))
        return keras.losses.categorical_crossentropy(y_true, y_pred)
    return f

def charbonnierLoss(y_true, y_pred):
    return K.mean(K.sqrt(K.square(y_pred - y_true) + K.constant(1e-6)),-1)

def unsupervised_distance_loss(y_fixed, gamma=1):
    # Pass in y_fixed, which is a set of n_classes x n_channels fixed points. Note that n_classes can be increased beyond the original # of classes if need be
    def loss(y_true, y_pred):
        # There are original B inputs, of form B x H x R x C
        # y_true will be N x C original values (same as batch_x from NeMO_generator)
        # y_pred will be N x C values predicted by the CNN

        # Let's see if this will work, or if y_true and y_pred HAVE to be exactly identical
        M = y_fixed.shape[0]
        n_channels = y_fixed.shape[1]
        N = y_pred.shape[0] # if y_pred is a tensor, N will be ?

        # Ky_fixed = K.repeat_elements(K.expand_dims(K.variable(y_fixed),axis=-1),N,-1)
        Ky_pred = K.repeat_elements(K.reshape(K.transpose(y_pred),(1,n_channels,-1)),M,0)
        Ky_fixed = K.expand_dims(K.variable(y_fixed),-1)

        dist = K.sqrt(K.sum(K.square(Ky_fixed-Ky_pred), axis=1, keepdims=True)) 
        sum_exp_dist_yf = K.sum(K.exp(-dist), axis=0, keepdims=True) # sum of exponential of distance across yt
        p_pred = 1/K.repeat_elements(sum_exp_dist_yf,M,0)*K.exp(-dist)
        p_pred = K.reshape(p_pred,(M,-1))
        dot_p_pred = K.dot(K.transpose(p_pred),p_pred)


        ky_true = K.reshape(K.transpose(y_true),(1,n_channels,-1))
        orig_dist = K.pow(K.sqrt(K.sum(K.square(K.expand_dims(y_true,-1)-ky_true), axis=1, keepdims=False)),gamma)
        toreturn = K.sum(dot_p_pred*orig_dist)
        return toreturn
    return loss

def lovasz_grad(gt_sorted):
    gts = tf.reduce_sum(gt_sorted)
    intersection = gts - tf.cumsum(gt_sorted)
    union = gts + tf.cumsum(1. - gt_sorted)
    jaccard = 1. - intersection / union
    jaccard = tf.concat((jaccard[0:1], jaccard[1:] - jaccard[:-1]), 0)
    return jaccard

def lovasz_softmax(probas, labels, classes='present', per_image=False, ignore=None, order='BHWC'):
    """
    Multi-class Lovasz-Softmax loss
      probas: [B, H, W, C] or [B, C, H, W] Variable, class probabilities at each prediction (between 0 and 1)
              Interpreted as binary (sigmoid) output with outputs of size [B, H, W].
      labels: [B, H, W] Tensor, ground truth labels (between 0 and C - 1)
      classes: 'all' for all, 'present' for classes present in labels, or a list of classes to average.
      per_image: compute the loss per image instead of per batch
      ignore: void class labels
      order: use BHWC or BCHW
    """
    if per_image:
        def treat_image(prob_lab):
            prob, lab = prob_lab
            prob, lab = tf.expand_dims(prob, 0), tf.expand_dims(lab, 0)
            prob, lab = flatten_probas(prob, lab, ignore, order)
            return lovasz_softmax_flat(prob, lab, classes=classes)
        losses = tf.map_fn(treat_image, (probas, labels), dtype=tf.float32)
        loss = tf.reduce_mean(losses)
    else:
        labels = tf.argmax(labels, axis=-1)
        loss = lovasz_softmax_flat(*flatten_probas(probas, labels, ignore, order), classes=classes)
    return loss

def lovasz_softmax_flat(probas, labels, classes='present'):
    """
    Multi-class Lovasz-Softmax loss
      probas: [P, C] Variable, class probabilities at each prediction (between 0 and 1)
      labels: [P] Tensor, ground truth labels (between 0 and C - 1)
      classes: 'all' for all, 'present' for classes present in labels, or a list of classes to average.
    """
    C = probas.shape[1]
    losses = []
    present = []
    class_to_sum = list(range(C)) if classes in ['all', 'present'] else classes
    for c in class_to_sum:
        fg = tf.cast(tf.equal(labels, c), probas.dtype)  # foreground for class c
        if classes == 'present':
            present.append(tf.reduce_sum(fg) > 0)
        if C == 1:
            if len(classes) > 1:
                raise ValueError('Sigmoid output possible only with 1 class')
            class_pred = probas[:, 0]
        else:
            class_pred = probas[:, c]
        errors = tf.abs(fg - class_pred)
        errors_sorted, perm = tf.nn.top_k(errors, k=tf.shape(errors)[0], name="descending_sort_{}".format(c))
        fg_sorted = tf.gather(fg, perm)
        grad = lovasz_grad(fg_sorted)
        losses.append(tf.tensordot(errors_sorted, tf.stop_gradient(grad), 1, name="loss_class_{}".format(c)))
    if len(class_to_sum) == 1:  # short-circuit mean when only one class
        return losses[0]
    losses_tensor = tf.stack(losses)
    if classes == 'present':
        present = tf.stack(present)
        losses_tensor = tf.boolean_mask(losses_tensor, present)
    loss = tf.reduce_mean(losses_tensor)
    return loss

def flatten_probas(probas, labels, ignore=None, order='BHWC'):
    """
    Flattens predictions in the batch
    """
    if len(probas.shape) == 3:
        probas, order = tf.expand_dims(probas, 3), 'BHWC'
    if order == 'BCHW':
        probas = tf.transpose(probas, (0, 2, 3, 1), name="BCHW_to_BHWC")
        order = 'BHWC'
    if order != 'BHWC':
        raise NotImplementedError('Order {} unknown'.format(order))
    C = probas.shape[3]
    probas = tf.reshape(probas, (-1, C))
    labels = tf.reshape(labels, (-1,))
    if ignore is None:
        return probas, labels
    valid = tf.not_equal(labels, ignore)
    vprobas = tf.boolean_mask(probas, valid, name='valid_probas')
    vlabels = tf.boolean_mask(labels, valid, name='valid_labels')
    return vprobas, vlabels

def keras_lovasz_softmax(labels,probas):
    #return lovasz_softmax(probas, labels)+binary_crossentropy(labels, probas)
    return lovasz_softmax(probas, labels)

def categorical_focal_loss(gamma=2.0, alpha=0.25):
    """
    Implementation of Focal Loss from the paper in multiclass classification
    Formula:
        loss = -alpha*((1-p)^gamma)*log(p)
    Parameters:
        alpha -- the same as wighting factor in balanced cross entropy
        gamma -- focusing parameter for modulating factor (1-p)
    Default value:
        gamma -- 2.0 as mentioned in the paper
        alpha -- 0.25 as mentioned in the paper
    """
    def focal_loss(y_true, y_pred):
        # Define epsilon so that the backpropagation will not result in NaN
        # for 0 divisor case
        epsilon = K.epsilon()
        # Add the epsilon to prediction value
        #y_pred = y_pred + epsilon
        # Clip the prediction value
        y_pred = K.clip(y_pred, epsilon, 1.0-epsilon)
        # Calculate cross entropy
        cross_entropy = -y_true*K.log(y_pred)
        # Calculate weight that consists of  modulating factor and weighting factor
        weight = alpha * y_true * K.pow((1-y_pred), gamma)
        # Calculate focal loss
        loss = weight * cross_entropy
        # Sum the losses in mini_batch
        loss = K.sum(loss, axis=1)
        return loss
    
    return focal_loss
