"""Pascal VOC Segmenttion Generator."""
from __future__ import unicode_literals
import os
import numpy as np
import multiprocessing.pool
import threading
import warnings
import loadcoraldata_utils as coralutils
from osgeo import gdal, ogr, osr
from functools import partial
from keras import backend as K
from keras.utils.np_utils import to_categorical
from glob import glob
from keras.preprocessing.image import (
    ImageDataGenerator,
    Iterator,
    load_img,
    img_to_array,
    pil_image,
    array_to_img,
    _count_valid_files_in_directory,
    _list_valid_filenames_in_directory)

from hyperopt import fmin, tpe, hp, STATUS_OK, Trials

class NeMOImageGenerator(ImageDataGenerator):
    """A real-time data augmentation generator for NeMO-Net Images"""

    def __init__(self,
                 image_shape=(100, 100, 3),
                 image_resample=True,
                 pixelwise_center=False,
                 pixel_mean=(0., 0., 0.),
                 pixelwise_std_normalization=False,
                 pixel_std=(1., 1., 1.),
                 featurewise_center=False,
                 samplewise_center=False,
                 featurewise_std_normalization=False,
                 samplewise_std_normalization=False,
                 zca_whitening=False,
                 rotation_range=0.,
                 width_shift_range=0.,
                 height_shift_range=0.,
                 shear_range=0.,
                 zoom_range=0.,
                 channel_shift_range=0.,
                 fill_mode='nearest',
                 cval=0.,
                 horizontal_flip=False,
                 vertical_flip=False,
                 rescale=None,
                 preprocessing_function=None,
                 data_format=None):
        """Init."""
        self.image_shape = tuple(image_shape)
        self.image_resample = image_resample
        self.pixelwise_center = pixelwise_center
        self.pixel_mean = np.array(pixel_mean)
        self.pixelwise_std_normalization = pixelwise_std_normalization
        self.pixel_std = np.array(pixel_std)
        super(NeMOImageGenerator, self).__init__()

    def standardize(self, x):
        """Standardize image."""
        if self.pixelwise_center:
            x -= self.pixel_mean
        if self.pixelwise_std_normalization:
            x /= self.pixel_std
        return super(NeMOImageGenerator, self).standardize(x) # If there are any other operations that needs to be performed on x in superclass

    def flow_from_imageset(self, image_set_loader,
                           class_mode='categorical', classes=None,
                           batch_size=1, shuffle=True, seed=None):
        """NeMOImageGenerator."""
        return IndexIterator(
            image_set_loader, self,
            class_mode=class_mode,
            classes=classes,
            batch_size=batch_size,
            shuffle=shuffle,
            seed=seed)

    def flow_from_NeMOdirectory(self, directory, FCN_directory=None,
                            target_size=(256, 256), color_mode='rgb',
                            classes=None, class_mode='categorical',
                            batch_size=32, class_weights = None, shuffle=True, seed=None,
                            save_to_dir=None,
                            save_prefix='',
                            save_format='png',
                            follow_links=False):
        return NeMODirectoryIterator(
            directory, self, FCN_directory=FCN_directory, target_size=target_size, color_mode=color_mode, classes=classes, class_mode=class_mode,
            data_format=self.data_format, batch_size=batch_size, class_weights=class_weights, shuffle=shuffle, seed=seed,
            save_to_dir=save_to_dir, save_prefix=save_prefix, save_format=save_format, follow_links=follow_links)

class IndexIterator(Iterator):
    """Iterator over index."""

    def __init__(self, image_set_loader, image_data_generator,
                 class_mode='categorical', classes=None,
                 batch_size=1, shuffle=False, seed=None, labelkey=[0,63,127,191]):
        """Init."""
        self.image_set_loader = image_set_loader
        self.image_data_generator = image_data_generator

        self.filenames = image_set_loader.filenames
        self.image_shape = image_set_loader.image_shape
        self.labelkey = labelkey

        self.classes = classes
        if class_mode == 'binary':
            label_shape = list(self.image_shape).pop(self.channel_axis - 1)
            self.label_shape = tuple(label_shape)
        elif class_mode == 'categorical':
            label_shape = list(self.image_shape)
            label_shape[self.image_data_generator.channel_axis - 1] \
                = self.classes
            self.label_shape = tuple(label_shape)

        super(IndexIterator, self).__init__(len(self.filenames), batch_size,
                                            shuffle, seed)

    def _get_batches_of_transformed_samples(self, index_array):
        batch_x = np.zeros(
            (self.batch_size,) + self.image_shape,
            dtype=K.floatx())
        batch_y = np.zeros(
            (self.batch_size,) + self.label_shape,
            dtype=np.int8)
        #batch_y = np.reshape(batch_y, (current_batch_size, -1, self.classes))

        for i, j in enumerate(index_array):
            fn = self.filenames[j]
            x = self.image_set_loader.load_img(fn)
            x = self.image_data_generator.standardize(x)
            batch_x[i] = x
            y = self.image_set_loader.load_seg(fn,labelkey=self.labelkey)
            y = to_categorical(y, self.classes).reshape(self.label_shape)
            #y = np.reshape(y, (-1, self.classes))
            batch_y[i] = y

        return batch_x, batch_y

class NeMODirectoryIterator(Iterator):
    """Iterator capable of reading images from a directory on disk.

    # Arguments
        directory: Path to the directory to read images from.
            Each subdirectory in this directory will be
            considered to contain images from one class,
            or alternatively you could specify class subdirectories
            via the `classes` argument.
        image_data_generator: Instance of `ImageDataGenerator`
            to use for random transformations and normalization.
        FCN_directory: Directory to use for FCN data, or complete labelled patch data
        target_size: tuple of integers, dimensions to resize input images to.
        color_mode: One of `"rgb"`, `"grayscale"`. Color mode to read images.
        classes: Optional list of strings, names of sudirectories
            containing images from each class (e.g. `["dogs", "cats"]`).
            It will be computed automatically if not set.
        class_mode: Mode for yielding the targets:
            `"binary"`: binary targets (if there are only two classes),
            `"categorical"`: categorical targets,
            `"sparse"`: integer targets,
            `"input"`: targets are images identical to input images (mainly
                used to work with autoencoders),
            `None`: no targets get yielded (only input images are yielded).
        batch_size: Integer, size of a batch.
        shuffle: Boolean, whether to shuffle the data between epochs.
        seed: Random seed for data shuffling.
        data_format: String, one of `channels_first`, `channels_last`.
        save_to_dir: Optional directory where to save the pictures
            being yielded, in a viewable format. This is useful
            for visualizing the random transformations being
            applied, for debugging purposes.
        save_prefix: String prefix to use for saving sample
            images (if `save_to_dir` is set).
        save_format: Format to use for saving sample images
            (if `save_to_dir` is set).
    """

    def __init__(self, directory, image_data_generator, FCN_directory=None,
                 target_size=(256, 256), color_mode='rgb',
                 classes=None, class_mode='categorical',
                 batch_size=32, class_weights=None, shuffle=True, seed=None,
                 data_format=None,
                 save_to_dir=None, save_prefix='', save_format='png',
                 follow_links=False):
        if data_format is None:
            data_format = K.image_data_format() #channels_last
        self.directory = directory
        self.FCN_directory = FCN_directory
        self.image_data_generator = image_data_generator
        self.target_size = tuple(target_size)
        if color_mode not in {'rgb', 'grayscale','8channel'}:
            raise ValueError('Invalid color mode:', color_mode,
                             '; expected "rgb", "grayscale", or "8channel.')
        self.color_mode = color_mode
        self.data_format = data_format
        if self.color_mode == 'rgb':
            if self.data_format == 'channels_last':
                self.image_shape = self.target_size + (3,)
            else:
                self.image_shape = (3,) + self.target_size
        elif self.color_mode == "8channel":
            if self.data_format == 'channels_last':
                self.image_shape = self.target_size + (8,)
            else:
                self.image_shape = (8,) + self.target_size
        else:
            if self.data_format == 'channels_last':
                self.image_shape = self.target_size + (1,)
            else:
                self.image_shape = (1,) + self.target_size
        if class_mode not in {'categorical', 'binary', 'sparse',
                              'input', None}:
            raise ValueError('Invalid class_mode:', class_mode,
                             '; expected one of "categorical", '
                             '"binary", "sparse", "input"'
                             ' or None.')
        self.class_mode = class_mode
        self.save_to_dir = save_to_dir
        self.save_prefix = save_prefix
        self.save_format = save_format
        self.class_weights = class_weights

        white_list_formats = {'png', 'jpg', 'jpeg', 'bmp', 'ppm', 'tif'}

        # first, count the number of samples and classes
        self.samples = 0

        if not classes:
            classes = []
            for subdir in sorted(os.listdir(directory)):
                if os.path.isdir(os.path.join(directory, subdir)):
                    classes.append(subdir)

        if type(classes) is dict:
            self.class_indices = classes
            self.num_consolclass = len(np.unique([self.class_indices[k] for k in self.class_indices]))
            classes = [k for k in self.class_indices] #redefine classes as a list
        else:
            self.class_indices = dict(zip(classes, range(len(classes))))
            self.num_consolclass = len(classes) # number of consolidated classes, which classes is a dictionary of
        self.num_class = len(classes) # sets num_class to TOTAL number of classes (NOT consolidate classes)

        pool = multiprocessing.pool.ThreadPool()
        function_partial = partial(_count_valid_files_in_directory,
                                   white_list_formats=white_list_formats,
                                   follow_links=follow_links)
        self.samples = sum(pool.map(function_partial,
                                    (os.path.join(directory, subdir)
                                     for subdir in classes)))
        print('Found %d images belonging to %d classes, split into %d consolidated classes.' % (self.samples, self.num_class, self.num_consolclass))

        # Check FCN label directory if specified
        if FCN_directory is not None:
            labelsamples = sum(pool.map(function_partial,
                (os.path.join(FCN_directory, subdir) for subdir in classes)))
            if labelsamples != self.samples:
                raise ValueError("Error! %d training images found but only %d labelled images found." %(self.samples,labelsamples))

        # second, build an index of the images in the different class subfolders
        results = []
        self.filenames = []
        self.classes = np.zeros((self.samples,), dtype='int32')
        i = 0
        for dirpath in (os.path.join(directory, subdir) for subdir in classes):
            results.append(pool.apply_async(_list_valid_filenames_in_directory,
                                            (dirpath, white_list_formats,
                                             self.class_indices, follow_links)))
        for res in results:
            tempclasses, filenames = res.get()
            self.classes[i:i + len(tempclasses)] = tempclasses
            self.filenames += filenames
            i += len(tempclasses)

        # Build an index of images in FCN label directory if specified
        if FCN_directory is not None:
            FCN_results = []
            self.FCN_filenames = []
            self.labelkey = [np.uint8(255/self.num_class*i) for i in range(self.num_class)]
            label_shape = list(self.image_shape)
            label_shape[self.image_data_generator.channel_axis - 1] = self.num_consolclass
            self.label_shape = tuple(label_shape)

            for dirpath in (os.path.join(FCN_directory, subdir) for subdir in classes):
                FCN_results.append(pool.apply_async(_list_valid_filenames_in_directory,
                    (dirpath, white_list_formats, self.class_indices, follow_links)))

            for res in FCN_results:
                tempclasses, filenames = res.get()
                self.FCN_filenames += filenames

        pool.close()
        pool.join()
        super(NeMODirectoryIterator, self).__init__(self.samples, batch_size, shuffle, seed) #n, batch_size, shuffle, seed

    def _flow_index(self, n, batch_size=32, shuffle=False, seed=None): #n = total number of images in all folders
        # Ensure self.batch_index is 0.
        self.reset()
        div = n/self.num_class
        while 1:
            index_array = []
            np.asarray(index_array)
            if seed is not None:
                np.random.seed(seed + self.total_batches_seen)

            leftover = batch_size % self.num_class #remainder in case num_class doesn't divide into batch_size
            random_c = np.random.choice(self.num_class,leftover,replace=False)
            for c in range(self.num_class):
                index_array = np.append(index_array, np.random.randint(int(np.round(c*div)),int(np.round(c*div+div)),size=int(batch_size//self.num_class)))
            for c in random_c: #choose random folders to pick from
                index_array = np.append(index_array, np.random.randint(int(np.round(c*div)),int(np.round(c*div+div)),size=1))

            current_index = (self.batch_index * batch_size) % n # remainder
            if n > current_index + batch_size:
                current_batch_size = batch_size
                self.batch_index += 1  # batch number within n images
            else:
                # current_batch_size = n - current_index
                current_batch_size = batch_size
                self.batch_index = 0 # has gone through all n images, reset batch_index to 0
            self.total_batches_seen += 1
            index_array = index_array.astype(np.int64)
            yield (index_array, current_index, current_batch_size)

    def next(self):
        """For python 2.x.

        # Returns
            The next batch.
        """
        with self.lock:
            index_array, current_index, current_batch_size = next(self.index_generator)
            # print("index array length: " + str(len(index_array)))
            # print("batch size: " + str(current_batch_size))
        # The transformation of images is not under thread lock
        # so it can be done in parallel
        batch_x = np.zeros((current_batch_size,) + self.image_shape, dtype=K.floatx())
        if self.color_mode == "8channel":
            for i, j in enumerate(index_array):
                fname = self.filenames[j]
                img = coralutils.CoralData(os.path.join(self.directory, fname), load_type="raster").image
                x = img_to_array(img, data_format=self.data_format)
                # x = self.image_data_generator.random_transform(x)
                x = self.image_data_generator.standardize(x)
                batch_x[i] = x
                # build batch of labels
            if self.class_mode == 'input':
                batch_y = batch_x.copy()
            elif self.class_mode == 'sparse':
                batch_y = self.classes[index_array]
            elif self.class_mode == 'binary':
                batch_y = self.classes[index_array].astype(K.floatx())
            elif self.FCN_directory is not None:
                batch_y = np.zeros((len(batch_x),) + self.label_shape, dtype=np.int8) # N x R x C x D
                batch_weights = np.zeros((len(batch_x), self.image_shape[0], self.image_shape[1]), dtype=np.float)
                for i, j in enumerate(index_array):
                    fname = self.FCN_filenames[j]
                    y = self._load_seg(fname)
                    if self.class_weights is not None:
                        weights = np.zeros((self.image_shape[0], self.image_shape[1]), dtype=np.float)
                        for k in self.class_weights:
                            weights[y == k] = self.class_weights[k]
                        batch_weights[i] = weights
                    y = to_categorical(y, self.num_consolclass).reshape(self.label_shape)
                    batch_y[i] = y

            elif self.class_mode == 'categorical':
                batch_y = np.zeros((len(batch_x), self.num_consolclass), dtype=K.floatx())
                for i, label in enumerate(self.classes[index_array]):
                    batch_y[i, label] = 1.
            else:
                return batch_x

        else:
            grayscale = self.color_mode == 'grayscale'
            # build batch of image data
            for i, j in enumerate(index_array):
                fname = self.filenames[j]
                img = load_img(os.path.join(self.directory, fname),
                               grayscale=grayscale,
                               target_size=self.target_size)
                x = img_to_array(img, data_format=self.data_format)
                x = self.image_data_generator.random_transform(x)
                x = self.image_data_generator.standardize(x)
                batch_x[i] = x
            # optionally save augmented images to disk for debugging purposes
            if self.save_to_dir:
                for i in range(current_batch_size):
                    img = array_to_img(batch_x[i], self.data_format, scale=True)
                    fname = '{prefix}_{index}_{hash}.{format}'.format(prefix=self.save_prefix,
                                                                      index=current_index + i,
                                                                      hash=index_array[i],
                                                                      format=self.save_format)
                    img.save(os.path.join(self.save_to_dir, fname))
            # build batch of labels
            if self.class_mode == 'input':
                batch_y = batch_x.copy()
            elif self.class_mode == 'sparse':
                batch_y = self.classes[index_array]
            elif self.class_mode == 'binary':
                batch_y = self.classes[index_array].astype(K.floatx())
            elif self.class_mode == 'categorical':
                batch_y = np.zeros((len(batch_x), self.num_class), dtype=K.floatx())
                for i, label in enumerate(self.classes[index_array]):
                    batch_y[i, label] = 1.
            else:
                return batch_x

        quick_shape1 = lambda z: np.reshape(z, (z.shape[0],z.shape[1]*z.shape[2],z.shape[3]))
        quick_shape2 = lambda z: np.reshape(z, (z.shape[0],z.shape[1]*z.shape[2]))

        if self.class_weights is None:
            return batch_x, batch_y
        else:
            return batch_x, quick_shape1(batch_y), quick_shape2(batch_weights)

    def _load_seg(self, fn):
        """Segmentation load method.

        # Arguments
            fn: filename of the image (with extension suffix)
        # Returns
            arr: numpy array of shape self.target_size
        """
        label_path = os.path.join(self.FCN_directory, fn)
        img = pil_image.open(label_path)
        if self.target_size:
            wh_tuple = (self.target_size[1], self.target_size[0])
        if img.size != wh_tuple:
            img = img.resize(wh_tuple)
        y = img_to_array(img, data_format=self.data_format)
        y = y.reshape(wh_tuple)

        item_counter = 0
        for item in self.labelkey:
            y[y == item] = self.class_indices[list(self.class_indices.keys())[item_counter]]
            item_counter+=1
        return y

    # def next(self):
    #     """Next batch."""
    #     with self.lock:
    #         index_array, current_index, current_batch_size = next(
    #             self.index_generator)

    #     return self._get_batches_of_transformed_samples(index_array)
    #     # batch_x = np.zeros(
        #     (current_batch_size,) + self.image_shape,
        #     dtype=K.floatx())
        # batch_y = np.zeros(
        #     (current_batch_size,) + self.label_shape,
        #     dtype=np.int8)
        # #batch_y = np.reshape(batch_y, (current_batch_size, -1, self.classes))

        # for i, j in enumerate(index_array):
        #     fn = self.filenames[j]
        #     x = self.image_set_loader.load_img(fn)
        #     x = self.image_data_generator.standardize(x)
        #     batch_x[i] = x
        #     y = self.image_set_loader.load_seg(fn,labelkey=labelkey)
        #     y = to_categorical(y, self.classes).reshape(self.label_shape)
        #     #y = np.reshape(y, (-1, self.classes))
        #     batch_y[i] = y

        # # save augmented images to disk for debugging
        # #if self.image_set_loader.save_to_dir:
        # #    for i in range(current_batch_size):
        # #        x = batch_x[i]
        # #        y = batch_y[i].argmax(
        # #            self.image_data_generator.channel_axis - 1)
        # #        if self.image_data_generator.data_format == 'channels_first':
        # #            y = y[np.newaxis, ...]
        # #        else:
        # #            y = y[..., np.newaxis]
        # #        self.image_set_loader.save(x, y, current_index + i)

        # return batch_x, batch_y


class ImageSetLoader(object):
    """Helper class to load image data into numpy arrays."""

    def __init__(self, image_set, image_dir, label_dir, target_size=(100, 100),
                 image_format='jpg', color_mode='rgb', label_format='png',
                 data_format=None,
                 save_to_dir=None, save_prefix='', save_format='jpg'):
        """Init."""
        if data_format is None:
            data_format = K.image_data_format()
        self.data_format = data_format
        self.target_size = tuple(target_size)

        if not os.path.exists(image_set):
            raise IOError('Image set {} does not exist. Please provide a'
                          'valid file.'.format(image_set))

        try:
            self.filenames = np.loadtxt(image_set, dtype=bytes)
            self.filenames = [fn.decode('utf-8') for fn in self.filenames]
        except:
            pass

        if not os.path.exists(image_dir):
            raise IOError('Directory {} does not exist. Please provide a '
                          'valid directory.'.format(image_dir))
        self.image_dir = image_dir
        if label_dir and not os.path.exists(label_dir):
            raise IOError('Directory {} does not exist. Please provide a '
                          'valid directory.'.format(label_dir))
        self.label_dir = label_dir

        white_list_formats = {'png', 'jpg', 'jpeg', 'bmp','tif'}
        self.image_format = image_format
        if self.image_format not in white_list_formats:
            raise ValueError('Invalid image format:', image_format,
                             '; expected "png", "jpg", "jpeg" or "bmp"')
        self.label_format = label_format
        if self.label_format not in white_list_formats:
            raise ValueError('Invalid image format:', label_format,
                             '; expected "png", "jpg", "jpeg" or "bmp"')

        if color_mode not in {'rgb', 'grayscale', '8channel'}:
            raise ValueError('Invalid color mode:', color_mode,
                             '; expected "rgb" or "grayscale".')
        self.color_mode = color_mode
        if self.color_mode == 'rgb':
            if self.data_format == 'channels_last':
                self.image_shape = self.target_size + (3,)
            else:
                self.image_shape = (3,) + self.target_size
        elif self.color_mode == '8channel':
            if self.data_format == 'channels_last':
                self.image_shape = self.target_size + (8,)
            else:
                self.image_shape = (8,) + self.target_size
        else:
            if self.data_format == 'channels_last':
                self.image_shape = self.target_size + (1,)
            else:
                self.image_shape = (1,) + self.target_size

        self.grayscale = self.color_mode == 'grayscale'

        self.save_to_dir = save_to_dir
        self.save_prefix = save_prefix
        self.save_format = save_format

    def load_img(self, fn):
        """Image load method.

        # Arguments
            fn: filename of the image (without extension suffix)
        # Returns
            arr: numpy array of shape self.image_shape
        """
        img_path = os.path.join(self.image_dir,
                                '{}.{}'.format(fn,
                                               self.image_format))
        if not os.path.exists(img_path):
            raise IOError('Image {} does not exist.'.format(img_path))
        img = load_img(img_path, self.grayscale, self.target_size)
        x = img_to_array(img, data_format=self.data_format)

        return x

    def load_seg(self, fn, labelkey=None):
        """Segmentation load method.

        # Arguments
            fn: filename of the image (without extension suffix)
        # Returns
            arr: numpy array of shape self.target_size
        """
        label_path = os.path.join(self.label_dir,
                                  '{}.{}'.format(fn, self.label_format))
        img = pil_image.open(label_path)
        if self.target_size:
            wh_tuple = (self.target_size[1], self.target_size[0])
        if img.size != wh_tuple:
            img = img.resize(wh_tuple)
        y = img_to_array(img, self.data_format)
 #       y[y == 255] = 0

        if labelkey is not None:
            item_counter = 0
            for item in labelkey:
                y[y == item ] = item_counter 
                item_counter+=1
        return y

    def save(self, x, y, index):
        """Image save method."""
        img = array_to_img(x, self.data_format, scale=True)
        mask = array_to_img(y, self.data_format, scale=True)
        img.paste(mask, (0, 0), mask)

        fname = 'img_{prefix}_{index}_{hash}.{format}'.format(
            prefix=self.save_prefix,
            index=index,
            hash=np.random.randint(1e4),
            format=self.save_format)
        img.save(os.path.join(self.save_to_dir, fname))
