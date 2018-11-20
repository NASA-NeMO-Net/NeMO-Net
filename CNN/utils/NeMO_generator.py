"""Pascal VOC Segmenttion Generator."""
from __future__ import unicode_literals
import os
import cv2
import numpy as np
import multiprocessing.pool
import threading
import warnings
import loadcoraldata_utils as coralutils
import random
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
from PIL import Image as pil_image

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
                 random_rotation=False,
                 fill_mode='nearest',
                 cval=0.,
                 horizontal_flip=False,
                 vertical_flip=False,
                 NeMO_rescale=None,
                 preprocessing_function=None,
                 data_format=None,
                 image_or_label="label"):
        """Init."""
        self.image_shape = tuple(image_shape)
        self.image_resample = image_resample
        self.pixelwise_center = pixelwise_center
        self.pixel_mean = np.array(pixel_mean)
        self.pixelwise_std_normalization = pixelwise_std_normalization
        self.pixel_std = np.array(pixel_std)
        self.NeMO_rescale = NeMO_rescale
        self.random_rotation = random_rotation
        self.image_or_label = image_or_label

        # Note that the below initialization mostly works with RGB data, use with care
        # However, rescale and channel_shift_range might be useful to multiply and shift individual channels, respectively
        super(NeMOImageGenerator, self).__init__(featurewise_center=featurewise_center, 
            samplewise_center = samplewise_center,
            featurewise_std_normalization = featurewise_std_normalization,
            samplewise_std_normalization=samplewise_std_normalization,
            zca_whitening = zca_whitening,
            rotation_range = rotation_range,
            width_shift_range = width_shift_range,
            height_shift_range = height_shift_range,
            shear_range = shear_range,
            zoom_range = zoom_range,
            channel_shift_range = channel_shift_range,
            fill_mode = fill_mode,
            cval = cval,
            horizontal_flip = horizontal_flip,
            vertical_flip = vertical_flip,
            preprocessing_function = preprocessing_function,
            data_format = data_format)

    def random_channel_shift(self, x):
        #Perform a random channel shift.
        # Arguments
            # x: Input tensor. Must be 3D.
        # Returns
            # Numpy image tensor.
        x = np.rollaxis(x, self.channel_axis-1, 0)
        min_x, max_x = -100, 100        # set arbitrarily large for now
        intensity = self.channel_shift_range
        if type(intensity) == float:
            channel_images = [np.clip(x_channel + np.random.uniform(0, intensity), min_x, max_x) for x_channel in x]    #Only one-directional randomization
        else:
            channel_images = [np.clip(x[i] + np.random.uniform(0, intensity[i]), min_x, max_x) for i in range(x.shape[0])]
        x = np.stack(channel_images, axis=0)
        x = np.rollaxis(x, 0, self.channel_axis)
        return x

    def random_flip_rotation(self,x, rnd_flip=None, rnd_rot=None):
        # Perform a random rotation (0, 90, 180, 270 deg)
        # Arguments
            # x: Input Image tensor (or label)
            # rnd_flip: Flip value (0 or 1), None = randomly decide
            # rnd_rot: # of times to rotate by 90 deg, None= randomly decide
        # Returns
            # x: Rotated image
            # rnd_flip: value of rnd_flip
            # rnd_rot: value of rnd_rot
        if rnd_flip is None:
            rnd_flip = np.random.randint(0,2)
        if rnd_rot is None:
            rnd_rot = np.random.randint(0,4)

        if rnd_flip:
            x = np.flip(x,0)

        x = np.rot90(x, rnd_rot)
        return x, rnd_flip, rnd_rot

    def standardize(self, x):
        # x is rows x cols x n_channels
        """Standardize image."""
        if self.NeMO_rescale is not None:
            rescale_params = [np.random.uniform(r[0],r[1]) for r in self.NeMO_rescale] # can only rescale to something smaller, maybe can consider a range later
            x *= rescale_params 
            x = np.clip(x, 0, self.pixel_mean*2) # pixel_mean might change later, but is currently 1/2 the max value
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
                            target_size=None, source_size=(64,64), color_mode='rgb',
                            passedclasses=None, class_mode='categorical',
                            batch_size=32, class_weights = None, shuffle=True, seed=None,
                            save_to_dir=None,
                            save_prefix='',
                            save_format='png',
                            follow_links=False,
                            image_or_label="label"):
        return NeMODirectoryIterator(
            directory, self, FCN_directory=FCN_directory, target_size=target_size, source_size=source_size, color_mode=color_mode, passedclasses=passedclasses, class_mode=class_mode,
            data_format=self.data_format, batch_size=batch_size, class_weights=class_weights, shuffle=shuffle, seed=seed,
            save_to_dir=save_to_dir, save_prefix=save_prefix, save_format=save_format, follow_links=follow_links, image_or_label=image_or_label)


# Currently unused (here from previous code)
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
        source_size: tuple of integers, dimensions of input images
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

    def __init__(self, directory, image_data_generator, FCN_directory=None, target_size=None, source_size=(64,64), color_mode='rgb',
                 passedclasses=None, class_mode='categorical', batch_size=32, class_weights=None, shuffle=True, seed=None,
                 data_format=None, save_to_dir=None, save_prefix='', save_format='png', follow_links=False, image_or_label="label"):
        if data_format is None:
            data_format = K.image_data_format() #channels_last
        if type(directory) == list:
            self.directory = directory
        else:
            self.directory = [directory]  # Make self.directory a list so we can iterate over it (matches it with if directory IS a list)
        self.FCN_directory = FCN_directory
        self.image_data_generator = image_data_generator
        self.source_size = tuple(source_size)

        if target_size is None:
            self.target_size = self.source_size
        else:
            self.target_size = tuple(target_size)

        if color_mode not in {'rgb', 'grayscale','8channel','4channel'}:
            raise ValueError('Invalid color mode:', color_mode,
                             '; expected "rgb", "grayscale", or "8channel.')
        self.color_mode = color_mode
        self.data_format = data_format
        
        if len(self.directory) == 1:
            source_size = [source_size]
        self.image_shape = []
        for dcount in range(0, len(self.directory)):
            if self.color_mode == 'rgb':
                if self.data_format == 'channels_last':
                    self.image_shape.append(self.source_size[dcount] + (3,))
                else:
                    self.image_shape.append((3,) + self.source_size[dcount])
            elif self.color_mode == "8channel":
                if self.data_format == 'channels_last':
                    self.image_shape.append(self.source_size[dcount] + (8,))
                else:
                    self.image_shape.append((8,) + self.source_size[dcount])
            elif self.color_mode == "4channel":
                if self.data_format == 'channels_last':
                    self.image_shape.append(self.source_size[dcount] + (4,))
                else:
                    self.image_shape.append((4,) + self.source_size[dcount])
            else:
                if self.data_format == 'channels_last':
                    self.image_shape.append(self.source_size[dcount] + (1,))
                else:
                    self.image_shape.append((1,) + self.source_size[dcount])
                            
        if class_mode not in {'categorical', 'binary', 'sparse', 'input', 'fixed_RGB', 'zeros', None}:
            raise ValueError('Invalid class_mode:', class_mode, '; expected one of "categorical", "binary", "sparse", "input", "fixed_RGB", "zeros", or None.')
        self.class_mode = class_mode
        self.save_to_dir = save_to_dir
        self.save_prefix = save_prefix
        self.save_format = save_format
        self.class_weights = class_weights
        self.image_or_label = image_or_label

        white_list_formats = {'png', 'jpg', 'jpeg', 'bmp', 'ppm', 'tif'}

        # first, count the number of samples and classes
        self.samples = 0

        # if not classes:
        # Always determine number classes by the subdirectories (will not count classes that don't show up)
        classes = [] 
        for subdir in sorted(os.listdir(self.directory[0])): #if multiple directories, make sure that they all have the same class folders
            if os.path.isdir(os.path.join(self.directory[0], subdir)):
                classes.append(subdir)

        if type(passedclasses) is dict:
            self.class_indices = passedclasses   # Class indices will be a dictionary containing the maximum possible classes, which is passed in
            self.num_consolclass = len(passedclasses) # number of consolidated classes, which passedclasses is a dictionary of
            # classes = [k for k in self.class_indices] #redefine classes as a list
        else:
            self.class_indices = dict(zip(classes, range(len(passedclasses)))) # Class indices determined from number of directories, USE THIS ONLY AS LAST RESORT IF YOU HAVE NO PASSABLE DICTIONARY
            self.num_consolclass = len(np.unique([self.class_indices[k] for k in self.class_indices]))
        self.num_class = len(classes) # sets num_class to number of existing subdirectory classes (NOT ALL consolidated classes)

        pool = multiprocessing.pool.ThreadPool()
        function_partial = partial(_count_valid_files_in_directory, white_list_formats=white_list_formats, follow_links=follow_links)
        
        for d in self.directory:
            self.samples = sum(pool.map(function_partial, (os.path.join(d, subdir) for subdir in classes))) # make sure number of samples same in both directories
            print('%s: Found %d images belonging to %d classes, split into %d consolidated classes.' % (d, self.samples, self.num_class, self.num_consolclass))

        # Check FCN label directory if specified
        if FCN_directory is not None:
            labelsamples = sum(pool.map(function_partial,
                (os.path.join(FCN_directory, subdir) for subdir in classes)))
            if labelsamples != self.samples:
                raise ValueError("Error! %d training images found but only %d labelled images found." %(self.samples,labelsamples))

        # second, build an index of the images in the different class subfolders
        results = []
        self.filenames = []
        self.classes = np.zeros((len(self.directory),self.samples,), dtype='int32')
        self.class_idx_startend = [] # for keeping track of the starting and ending idx of each class, in case they are of different lengths 
        dcount = 0
        for d in self.directory:
            tempresults = []
            for dirpath in (os.path.join(d, subdir) for subdir in classes):
                tempresults.append(pool.apply_async(_list_valid_filenames_in_directory, (dirpath, white_list_formats, self.class_indices, follow_links)))
            results.append(tempresults)
            
            i = 0
            tempfilenames = []
            for res in tempresults:
                tempclasses, filenames = res.get()
                self.classes[dcount,i:i + len(tempclasses)] = tempclasses
                if dcount == 0: # assumes that all files have corresponding 1:1 between inputs
                    self.class_idx_startend.append([i,i+len(tempclasses)])
                filenames.sort()
                tempfilenames += filenames
                i += len(tempclasses)
            self.filenames.append(tempfilenames)
            dcount += 1
            
        # Build an index of images in FCN label directory if specified
        if FCN_directory is not None:
            FCN_results = []
            self.FCN_filenames = []
            self.min_labelkey = np.min([self.class_indices[k] for k in self.class_indices])
            self.labelkey = [np.uint8(255/self.num_consolclass*i) for i in range(self.min_labelkey, self.min_labelkey+self.num_consolclass)] # Assuming labels are saved according to # of consolclass
            label_shape = list(self.image_shape[0]) # R x C x D
            if self.image_or_label == "label":
                label_shape[self.image_data_generator.channel_axis - 1] = self.num_consolclass
                self.label_shape = tuple(label_shape)
            else:
                label_shape = self.target_size + (self.image_shape[0][-1],)
                self.label_shape = tuple(label_shape)

            for dirpath in (os.path.join(FCN_directory, subdir) for subdir in classes):
                FCN_results.append(pool.apply_async(_list_valid_filenames_in_directory,
                    (dirpath, white_list_formats, self.class_indices, follow_links)))

            for res in FCN_results:
                tempclasses, filenames = res.get()
                filenames.sort()
                self.FCN_filenames += filenames
        pool.close()
        pool.join()
        super(NeMODirectoryIterator, self).__init__(self.samples, batch_size, shuffle, seed) #n, batch_size, shuffle, seed

    def _flow_index(self, n, batch_size=32, shuffle=False, seed=None): #n = total number of images in all folders
        # Ensure self.batch_index is 0.
        self.reset()
        # This assumes same number of files per class!
        div = n/self.num_class
        while 1:
            index_array = []
            np.asarray(index_array)
            if seed is not None:
                np.random.seed(seed + self.total_batches_seen)

            leftover = batch_size % self.num_class #remainder in case num_class doesn't divide into batch_size
            random_c = np.random.choice(self.num_class,leftover,replace=False)
            # Chooses same number of images per class, starting and ending with class_idx_startend in case folders contain different # of files
            for c in range(self.num_class):
                index_array = np.append(index_array, np.random.randint(int(self.class_idx_startend[c][0]),int(self.class_idx_startend[c][1]),size=int(batch_size//self.num_class)))
            for c in random_c: #choose random folders to pick from
                index_array = np.append(index_array, np.random.randint(int(self.class_idx_startend[c][0]),int(self.class_idx_startend[c][1]),size=1))

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
        # The transformation of images is not under thread lock so it can be done in parallel
        
        batch_x = [] # create batch as list, so it can support multiple inputs
        for i in range(len(self.directory)):
            batch_x.append(np.zeros((current_batch_size,) + self.image_shape[i], dtype=K.floatx()))
        
        if self.image_data_generator.random_rotation:
            batch_flip = np.zeros(current_batch_size)
            batch_rot90 = np.zeros(current_batch_size)
        if self.color_mode == "8channel" or self.color_mode == "4channel":
            # iterate over directory?
            for dcount in range(0,len(self.directory)):
                for i, j in enumerate(index_array):
                    fname = self.filenames[dcount][j]
                    # print('image filename: ', fname, j)
                    img = coralutils.CoralData(os.path.join(self.directory[dcount], fname), load_type="raster").image
                    if self.color_mode == "4channel": # assumes target AND source are starting off as 8 channel
                        img = np.delete(img, [0,3,5,7], 2) # harded coded for BGR + NIR
                    x = img_to_array(img, data_format=self.data_format)
                    x = self.image_data_generator.standardize(x) # standardize and rescaling is done here
                    if self.image_data_generator.channel_shift_range != 0:
                        x = self.image_data_generator.random_channel_shift(x)   
                    if self.image_data_generator.random_rotation:
                        if dcount == 0: # random rotation if first time, otherwise use same as before
                            x, batch_flip[i], batch_rot90[i] = self.image_data_generator.random_flip_rotation(x)
                        else:
                            x, _, _ = self.image_data_generator.random_flip_rotation(x, batch_flip[i], batch_rot90[i])
                    # print("3:",x.shape)
                    batch_x[dcount][i] = x
                    if self.save_to_dir: # save to directory code
                        img = (x*self.image_data_generator.pixel_std)+self.image_data_generator.pixel_mean
                        fname = '{prefix}_{direct}_{index}_{hash}.{format}'.format(prefix=self.save_prefix+'_trainimg', direct=dcount,
                                                                              index=current_index + i,
                                                                              hash=index_array[i],
                                                                              format='tif')
                        driver = gdal.GetDriverByName('GTiff')
                        dataset = driver.Create(os.path.join(self.save_to_dir, fname), img.shape[0], img.shape[1], img.shape[2], gdal.GDT_Float32)
                        for chan in range(img.shape[2]):
                            dataset.GetRasterBand(chan+1).WriteArray((img[:,:,chan]))
                            dataset.FlushCache()

                # build batch of labels
            if self.class_mode == 'input':
                batch_y = batch_x.copy()
            elif self.class_mode == 'sparse':
                batch_y = self.classes[index_array]
            elif self.class_mode == 'binary':
                batch_y = self.classes[index_array].astype(K.floatx())
            elif self.FCN_directory is not None:
                if self.image_or_label == "image":
                    batch_y = np.zeros((len(batch_x),) + self.label_shape, dtype=K.floatx()) # N x R x C x D
                elif self.image_or_label == "label":
                    batch_y = np.zeros((len(batch_x),) + self.label_shape, dtype=np.int8) # N x R x C x D
                batch_weights = np.zeros((len(batch_x), self.image_shape[0][0], self.image_shape[0][1]), dtype=np.float)
                
                for i, j in enumerate(index_array):
                    fname = self.FCN_filenames[j]
                    # print("FCN filename: ", fname, j)
                    
                    if self.image_or_label == "label":
                        y = self._load_seg(fname)
                        if self.image_data_generator.random_rotation:           # flip and rotate according to previous batch_x images
                            y, _, _ = self.image_data_generator.random_flip_rotation(y, batch_flip[i], batch_rot90[i])
                        if self.class_weights is not None:
                            weights = np.zeros((self.image_shape[0][0], self.image_shape[0][1]), dtype=np.float)
                            for k in self.class_weights:
                                weights[y == self.class_indices[k]] = self.class_weights[k]             #class_weights must be a dictionary
                            batch_weights[i] = weights
                        if self.save_to_dir:            # save to dir before y is transformed to categorical tensor
                            img = y
                            fname = '{prefix}_{index}_{hash}.{format}'.format(prefix=self.save_prefix+'_labelimg',
                                                                                  index=current_index + i,
                                                                                  hash=index_array[i],
                                                                                  format=self.save_format)
                            cv2.imwrite(os.path.join(self.save_to_dir, fname), img)
                        y = to_categorical(y, self.num_consolclass).reshape(self.label_shape)
                        batch_y[i] = y
                    elif self.image_or_label == "image":
                        img = coralutils.CoralData(os.path.join(self.FCN_directory, fname), load_type="raster").image # if image and 8channel, then this must also be 8channel
                        if self.color_mode == "4channel":
                            img = np.delete(img,[0,3,5,7], 2) # hard coded for BGR + NIR
                        y = img_to_array(img, data_format=self.data_format)
                        if self.image_data_generator.random_rotation:           # flip and rotate according to previous batch_x images
                            y, _, _ = self.image_data_generator.random_flip_rotation(y, batch_flip[i], batch_rot90[i])
                        if self.save_to_dir:            # save to dir before y is transformed to categorical tensor
                            img = y
                            fname = '{prefix}_{index}_{hash}.{format}'.format(prefix=self.save_prefix+'_labelimg',
                                                                                  index=current_index + i,
                                                                                  hash=index_array[i],
                                                                                  format='tif')
                            FCNdriver = gdal.GetDriverByName('GTiff')
                            FCNdataset = FCNdriver.Create(os.path.join(self.save_to_dir, fname), img.shape[0], img.shape[1], img.shape[2], gdal.GDT_Float32)
                            for chan in range(img.shape[2]):
                                FCNdataset.GetRasterBand(chan+1).WriteArray((img[:,:,chan]))
                                FCNdataset.FlushCache()
                        y = self.image_data_generator.standardize(y) # standardize and rescaling is done here
                        if self.image_data_generator.channel_shift_range != 0:
                            y = self.image_data_generator.random_channel_shift(y)  
                        batch_y[i] = y
            elif self.class_mode == 'zeros':
                batch_y = np.zeros((current_batch_size,) + self.target_size, dtype=K.floatx())
            elif self.class_mode == 'categorical':
                batch_y = np.zeros((current_batch_size, self.num_consolclass), dtype=K.floatx())
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
                               target_size=self.source_size)
                x = img_to_array(img, data_format=self.data_format)
                x = self.image_data_generator.standardize(x) # standardize and rescaling is done here
                if self.image_data_generator.channel_shift_range != 0:
                    x = self.image_data_generator.random_channel_shift(x)  
                batch_x[i] = x
            # optionally save augmented images to disk for debugging purposes
            if self.save_to_dir:
                for i in range(current_batch_size):
                    img = (batch_x[i]*self.image_data_generator.pixel_std)+self.image_data_generator.pixel_mean
                    img = pil_image.fromarray(img.astype('uint8'), 'RGB')
                    # img = array_to_img(batch_x[i], self.data_format, scale=True)
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
            elif self.FCN_directory is not None:
                if self.image_or_label == "image":
                    batch_y = np.zeros((len(batch_x),) + self.label_shape, dtype=K.floatx()) # N x R x C x D
                elif self.image_or_label == "label":
                    batch_y = np.zeros((len(batch_x),) + self.label_shape, dtype=np.int8) # N x R x C x D

                batch_weights = np.zeros((len(batch_x), self.image_shape[0], self.image_shape[1]), dtype=np.float)
                for i, j in enumerate(index_array):
                    fname = self.FCN_filenames[j]
                    if self.image_or_label == "label":
                        y = self._load_seg(fname)
                        if self.image_data_generator.random_rotation:           # flip and rotate according to previous batch_x images
                            y, _, _ = self.image_data_generator.random_flip_rotation(y, batch_flip[i], batch_rot90[i])
                        if self.class_weights is not None:
                            weights = np.zeros((self.image_shape[0], self.image_shape[1]), dtype=np.float)
                            for k in self.class_weights:
                                weights[y == self.class_indices[k]] = self.class_weights[k]             #class_weights must be a dictionary
                            batch_weights[i] = weights
                        if self.save_to_dir:            # save to dir before y is transformed to categorical tensor
                            img = y
                            img = pil_image.fromarray(img.astype('uint8'), 'RGB')
                            fname = '{prefix}_{index}_{hash}.{format}'.format(prefix=self.save_prefix+'_labelimg',
                                                                                  index=current_index + i,
                                                                                  hash=index_array[i],
                                                                                  format=self.save_format)
                            cv2.imwrite(os.path.join(self.save_to_dir, fname), img)
                        y = to_categorical(y, self.num_consolclass).reshape(self.label_shape)
                        batch_y[i] = y
                    elif self.image_or_label == "image":
                        img = load_img(os.path.join(self.FCN_directory, fname), grayscale=grayscale, target_size=self.target_size)
                        y = img_to_array(img, data_format=self.data_format)
                        if self.image_data_generator.random_rotation:           # flip and rotate according to previous batch_x images
                            y, _, _ = self.image_data_generator.random_flip_rotation(y, batch_flip[i], batch_rot90[i])
                        if self.save_to_dir:            # save to dir before y is transformed to categorical tensor
                            img = y
                            img = pil_image.fromarray(img.astype('uint8'), 'RGB')
                            fname = '{prefix}_{index}_{hash}.{format}'.format(prefix=self.save_prefix+'_labelimg',
                                                                                  index=current_index + i,
                                                                                  hash=index_array[i],
                                                                                  format=self.save_format)
                            img.save(os.path.join(self.save_to_dir, fname))
                        y = self.image_data_generator.standardize(y) # standardize and rescaling is done here
                        if self.image_data_generator.channel_shift_range != 0:
                            y = self.image_data_generator.random_channel_shift(y)  
                        batch_y[i] = y
            elif self.class_mode == 'categorical':
                batch_y = np.zeros((len(batch_x), self.num_consolclass), dtype=K.floatx())
                for i, label in enumerate(self.classes[index_array]):
                    batch_y[i, label] = 1.
            elif self.class_mode == 'fixed_RGB':        # Testing for skew comparisons (ignore)
                # assert self.FCN_directory is not None
                # Temporarily pass in FCN_directory as fixed RGB values of size [N x N_channels]
                # batch_y = self.FCN_directory  # probably want to rethink this
                midpoint = int(self.source_size[0]-1/2)
                batch_y = np.zeros((len(batch_x), 3), dtype=K.floatx())
                for i, label in enumerate(self.classes[index_array]):
                    # print("batch_x shape: ", batch_x.shape, midpoint)
                    batch_y[i] = batch_x[i,midpoint,midpoint,:]   # batch_x has already been standardized

            else:
                return batch_x

        if len(batch_x) == 1: # Check if there is only one directory
            batch_x = batch_x[0]
            
        quick_shape1 = lambda z: np.reshape(z, (z.shape[0],z.shape[1]*z.shape[2],z.shape[3])) # convert to B x (C*R) x N_channel
        quick_shape2 = lambda z: np.reshape(z, (z.shape[0],z.shape[1]*z.shape[2]))

        if self.image_or_label == "image" or self.image_or_label == "unaltered":
            return batch_x, batch_y
        else:
            if self.class_weights is None:
                return batch_x, quick_shape1(batch_y)
            else:
                return batch_x, quick_shape1(batch_y), quick_shape2(batch_weights)

    def _load_seg(self, fn):
        """Segmentation load method.

        # Arguments
            fn: filename of the image (with extension suffix)
        # Returns
            arr: numpy array of shape self.source_size
        """
        label_path = os.path.join(self.FCN_directory, fn)
        if label_path.endswith('.tif'):
            img = pil_image.open(label_path)
            if self.source_size:
                wh_tuple = (self.source_size[1], self.source_size[0])
            if img.size != wh_tuple:
                img = img.resize(wh_tuple)
            y = img_to_array(img, data_format=self.data_format)
            y = y.reshape(wh_tuple)
        elif label_path.endswith('.png'):
            img = cv2.imread(label_path, cv2.IMREAD_GRAYSCALE)
            y = img
        else:
            print('Unrecognized file format for _load_seg!')
            raise ValueError

        item_counter = 0
        # print(np.unique(y))
        # print(np.unique(self.labelkey))
        for item in self.labelkey:
            y[y == item] = self.class_indices[list(self.class_indices.keys())[item_counter]] - self.min_labelkey # Set 0-255 gray level to 0-num_consolclass
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
