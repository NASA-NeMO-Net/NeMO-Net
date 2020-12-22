from __future__ import unicode_literals
from typing import Tuple, Callable, List, Union

import numpy as np
import os

from NeMO_DirectoryIterator import NeMODirectoryIterator
from NeMO_Augmentation import PolynomialAugmentation
from NeMO_Utils import apply_channel_corrections, normalize

from keras.preprocessing.image import (
    ImageDataGenerator)

class NeMOImageGenerator(ImageDataGenerator):
    """ Image Generator for NeMO-Net

    # Arguments
        image_shape: Shape of input image. 
        pixel_mean: Mean value to subtract during normalization per channel. Note that train images may already be normalized with (mean,std) = (100,100).
            This would add further normalization per channel. None implies a mean of 0
        pixel_std: Std value to normalize per channel. Note that train images may already be normalized with (mean,std) = (100,100).
            This would add further normalization per channel. None implies a std of 1
        pre_pixel_mean: Mean value of normalization that was already applied to image prior to load. Default is 100
        pre_pixel_std: Std value of normalization that was already applied to image prior to load. Default is 100
        channel_shift_range: Range for random channel shifts (passed into keras.preprocessing.image.ImageDataGenerator)
        spectral_augmentation: Enable specifically derived spectral augmentation
        random_rotation: Enable random flips and 90/180/270 degree rotations of image
        preprocessing_function: function that will be applied on each input. The function will run after the image is resized and augmented. 
            The function should take one argument: one image (Numpy tensor with rank 3), and should output a Numpy tensor with the same shape.
    """

    def __init__(self,
                 image_shape: Tuple[int, int, int] = (100, 100, 3),
                 pixel_mean: Union[float, List[float]] = None,
                 pixel_std: Union[float, List[float]] = None,
                 pre_pixel_mean: Union[float, List[float]] = 100,
                 pre_pixel_std: Union[float, List[float]] = 100,
                 channel_shift_range: float = 0.,
                 spectral_augmentation: bool = True,
                 random_rotation: bool = False,
                 preprocessing_function: Callable = None):

        self.image_shape = tuple(image_shape)
        self.pixel_mean = apply_channel_corrections(pixel_mean, self.image_shape[2], 0.0, "pixel_mean")
        self.pixel_std = apply_channel_corrections(pixel_std, self.image_shape[2], 1.0, "pixel_std")
        self.spectral_augmentation = spectral_augmentation
        if spectral_augmentation:
            self.augmentation_module = PolynomialAugmentation(self.image_shape[2], 100.0, 100.0, reverse_normalize = True)
        self.random_rotation = random_rotation

        # Note that the below initialization mostly works with RGB data, so most of it's values are hard set to work with NeMO-Net Data
        # However, rescale and channel_shift_range might be useful to multiply and shift individual channels, respectively
        super(NeMOImageGenerator, self).__init__(featurewise_center = False, 
            samplewise_center = False,
            featurewise_std_normalization = False,
            samplewise_std_normalization = False,
            zca_whitening = False,
            rotation_range = 0.,
            width_shift_range = 0.,
            height_shift_range = 0.,
            shear_range = 0.,
            zoom_range = 0.,
            channel_shift_range = channel_shift_range,
            fill_mode = 'nearest',
            cval = 0.0,
            horizontal_flip = False,
            vertical_flip = False,
            preprocessing_function = preprocessing_function,
            data_format = "channels_last")

    def flow_from_NeMOdirectory(self, **kwargs) -> NeMODirectoryIterator:
        return NeMODirectoryIterator(self, **kwargs)

    def generator_normalize(self, 
        input_array: np.ndarray,
        pixel_mean: np.ndarray,
        pixel_std: np.ndarray,
        reverse_normalize: bool = False) -> np.ndarray:
        """
        Applies normalization  to input array based upon mean and std
            input_array: rows x cols x n_channels array to normalize
            pixel_mean: mean value PER channel to normalize with (use apply_channel_corrections to get this if necessary)
            pixel_std: std value PER channel to normalize with (use apply_channel_corrections to get this if necessary)
            reverse_normalize: reverse normalization procedure
        """
        array = normalize(input_array, pixel_mean, pixel_std, reverse_normalize)
        return super(NeMOImageGenerator, self).standardize(array) # If there are any other operations that needs to be performed

class ImageSetLoader(object):
    """Helper class to load image data into numpy arrays."""

    def __init__(self, 
        image_dir: str, 
        label_dir: str, 
        image_size: Tuple[int, int] = (256, 256),
        image_format: str = 'tif', 
        color_mode: str = '8channel', 
        label_format: str ='png'):

        self.image_size = tuple(image_size)

        if not os.path.exists(image_dir):
            raise IOError('Directory {} does not exist. Please provide a valid directory.'.format(image_dir))
        self.image_dir = image_dir

        if label_dir and not os.path.exists(label_dir):
            raise IOError('Directory {} does not exist. Please provide a valid directory.'.format(label_dir))
        self.label_dir = label_dir

        white_list_formats = {'png', 'jpg', 'jpeg', 'bmp','tif'}
        self.image_format = image_format
        if self.image_format not in white_list_formats:
            raise ValueError('Invalid image format:', image_format, '; expected "png", "jpg", "jpeg" or "bmp"')
        self.label_format = label_format
        if self.label_format not in white_list_formats:
            raise ValueError('Invalid image format:', label_format, '; expected "png", "jpg", "jpeg" or "bmp"')

        if color_mode not in {'rgb', 'grayscale', '8channel', '4channel'}:
            raise ValueError('Invalid color mode:', color_mode, '; expected "rgb" or "grayscale".')
        self.color_mode = color_mode

        self.data_format = 'channels_last'
        if self.color_mode == 'rgb':
            self.image_shape = self.image_size + (3,)
        elif self.color_mode == '8channel':
            self.image_shape = self.image_size + (8,)
        elif self.color_mode == '4channel':
            self.image_shape = self.image_size + (4,)
        else:
            self.image_shape = self.image_size + (1,)

        self.grayscale = self.color_mode == 'grayscale'