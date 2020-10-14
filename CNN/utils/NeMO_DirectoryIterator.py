import os
import multiprocessing.pool
import numpy as np
import cv2
from typing import Tuple, Dict
from functools import partial
from osgeo import gdal, ogr, osr

from NeMO_Augmentation import NeMOAugmentationModule, PolynomialAugmentation
from NeMO_Utils import apply_channel_corrections, normalize
import loadcoraldata_utils as coralutils

from keras import backend as K
from keras.utils.np_utils import to_categorical
from keras.preprocessing.image import (
    ImageDataGenerator,
    Iterator,
    load_img,
    img_to_array,
    pil_image,
    array_to_img,
    _count_valid_files_in_directory,
    _list_valid_filenames_in_directory)

class NeMODirectoryIterator(Iterator):
    """Iterator capable of reading images from a directory on disk.

    # Arguments
        image_data_generator: Instance of `ImageDataGenerator` to use for random transformations and normalization.
        directory: Path to the directory to read images from. Each subdirectory in this directory will be
            considered to contain images from one class. However, for semantically segmented data, this may not be accurate
            (since each image can contain multiple classes)
        label_directory: Directory to use for FCN data, or complete labelled patch data
        classes: Dictionary of classes and associated categorical value
        class_weights: Dictionary of classes and their associated weights
        class_mode: Mode for yielding the targets:
            "categorical": categorical targets
            "zeros": All zero outputs
            "image": image target
            None: no targets get yielded (only input images are yielded).
        image_size: tuple of integers, dimensions of input images
        color_mode: Type of satellite image data
            "rgb": RGB 3-channel
            "grayscale": 1-channel grayscale
            "8channel": 8 channel (standard for WV2)
            "4channel": 4 channel (standard for Sentinel)
            "8channel_to_4channel": Originally 8 channel, shortened to Sentinel 4 channel (with corresponding channels delete hard-coded) 
        batch_size: Integer, size of a batch.
        shuffle: Boolean, whether to shuffle the data between epochs.
        seed: Random seed for data shuffling.
        save_to_dir: Optional directory where to save the pictures being yielded, in a viewable format. This is useful
            for visualizing the random transformations being applied, for debugging purposes.
        save_prefix: String prefix to use for saving sample
            images (if `save_to_dir` is set).
        save_format: Format to use for saving sample images
            (if `save_to_dir` is set).
        reshape: Whether to reshape the output for in temporal state (useful for using weight training for semantically segmented data)
    """

    def __init__(self,
        image_data_generator,
        directory: str,
        label_directory: str,
        classes: Dict[str, int],
        class_weights: Dict[str, float] = None,
        class_mode: str = 'categorical',
        image_size: Tuple[int, int] = (256, 256), 
        color_mode: str = '8channel',
        batch_size: int = 32,
        shuffle: bool = True,
        seed: int = None,
        save_to_dir: str = None, 
        save_prefix: str = "",
        save_format: str = "png",
        reshape: bool = True):

        self.data_format = K.image_data_format() #channels_last

        self.directory = directory
        self.label_directory = label_directory
        self.image_data_generator = image_data_generator

        self.image_size = tuple(image_size)
        self.color_mode = color_mode
        if self.color_mode == 'rgb':
            self.image_shape = self.image_size + (3,)
        elif self.color_mode == "8channel":
            self.image_shape = self.image_size + (8,)
        elif self.color_mode == "4channel" or self.color_mode == "8channel_to_4channel":
            self.image_shape = self.image_size + (4,)
        else:
            self.image_shape = self.image_size + (1,)
                        
        if color_mode not in {'rgb', 'grayscale','8channel','4channel','8channel_to_4channel'}:
            raise ValueError('Invalid color mode:', color_mode, '; expected "rgb", "grayscale", "8channel", "4channel", or "8channel_to_4channel"')
        self.color_mode = color_mode    
        if class_mode not in {'categorical', 'image', 'zeros', None}:
            raise ValueError('Invalid class_mode:', class_mode, '; expected one of "categorical", image", "zeros", or None.')
        self.class_mode = class_mode

        self.save_to_dir = save_to_dir
        self.save_prefix = save_prefix
        self.save_format = save_format
        self.class_weights = class_weights
        self.reshape = reshape

        white_list_formats = {'png', 'jpg', 'jpeg', 'bmp', 'ppm', 'tif'}

        # count the number of samples and classes    
        subdir_classes = [] 
        for subdir in sorted(os.listdir(self.directory)):
            if os.path.isdir(os.path.join(self.directory, subdir)):
                subdir_classes.append(subdir) # Note that subdir classes are from subfolders, while passed classes actually will entail the # of classes overall
        self.num_subdir_classes = len(subdir_classes) #number of existing subdirectory classes (NOT passed classes)

        self.classes_dict = classes   # Class indices will be a dictionary containing the maximum possible classes, which is passed in
        self.num_classes = len(self.classes_dict) # number of consolidated classes, which passedclasses is a dictionary of

        pool = multiprocessing.pool.ThreadPool()
        function_partial = partial(_count_valid_files_in_directory, 
            white_list_formats = white_list_formats, 
            follow_links = False)
        
        # count number of sample in training images folder (including all subdirectories)
        self.samples = sum(pool.map(function_partial, (os.path.join(self.directory, subdir) for subdir in subdir_classes))) 
        print('%s: Found %d images belonging to %d classes, split into %d subdirectory classes.' % (self.directory, self.samples, self.num_classes, self.num_subdir_classes))

        # Make sure same number of images in label directory
        labelsamples = sum(pool.map(function_partial, (os.path.join(label_directory, subdir) for subdir in subdir_classes)))
        if labelsamples != self.samples:
            raise ValueError("Error! %d training images found but only %d labelled images found." %(self.samples, labelsamples))

        # Build an index of the images in the different class subfolders
        results = []
        self.class_idx_startend = [] # for keeping track of the starting and ending idx of each class, in case they are of different lengths 

        # find classes and filenames per subdirectory
        classes_and_filenames = []
        for dirpath in (os.path.join(self.directory, subdir) for subdir in subdir_classes):
            classes_and_filenames.append(pool.apply_async(_list_valid_filenames_in_directory, 
                (dirpath, white_list_formats, self.classes_dict, False)))

        # Organize classes and filenames in subdirectories into a list, so we can randomize over it later
        i = 0
        self.filenames = []
        self.subdir_classes_idx = np.zeros((self.samples,), dtype='int32') 
        for res in classes_and_filenames:
            res_classes, res_filenames = res.get()
            self.subdir_classes_idx[i:i + len(res_classes)] = res_classes
            self.class_idx_startend.append([i, i + len(res_classes)])
            res_filenames.sort()
            self.filenames += res_filenames
            i += len(res_classes)
        for count, subdir in enumerate(subdir_classes):
            print("Class index start and end for '%s' subdirectory: " % (subdir), self.class_idx_startend[count])

        if self.class_mode == 'categorical':
            self.label_shape = self.image_size + (self.num_classes,)
        elif self.class_mode == 'image':
            self.label_shape = self.image_shape
        elif self.class_mode == 'zeros':
            self.label_shape = self.image_size = (1,)
        else:
            self.label_shape = None

        # Build an index of label images (similar to training images)
        label_results = []
        self.label_filenames = []
        self.min_labelkey = np.min(list(self.classes_dict.values()))
        self.labelkey = {}
        for k, v in self.classes_dict.items():
            self.labelkey[k] = np.uint(255/self.num_classes * v)  

        for dirpath in (os.path.join(label_directory, subdir) for subdir in subdir_classes):
            label_results.append(pool.apply_async(_list_valid_filenames_in_directory,
                (dirpath, white_list_formats, self.classes_dict, False)))

        for res in label_results:
            tempclasses, filenames = res.get()
            filenames.sort()
            self.label_filenames += filenames
        pool.close()
        pool.join()
        super(NeMODirectoryIterator, self).__init__(self.samples, batch_size, shuffle, seed) #n, batch_size, shuffle, seed

    def _flow_index(self, n: int, batch_size: int = 32, shuffle: bool = False, seed: int = None): 
        # n = total number of images in all folders
        # Ensure self.batch_index is 0.
        self.reset()
        # This assumes same number of files per class!
           
        while 1:
            index_array = []

            if seed is not None:
                np.random.seed(seed + self.total_batches_seen)

            leftover = batch_size % self.num_subdir_classes #remainder in case num_class doesn't divide into batch_size
            random_c = np.random.choice(self.num_subdir_classes, leftover, replace=False)
            for c in range(self.num_subdir_classes):
                index_array.append(np.random.randint(int(self.class_idx_startend[c][0]), 
                    int(self.class_idx_startend[c][1]),
                    size = int(batch_size//self.num_subdir_classes)))
            for c in random_c: #choose random folders to pick from
                index_array.append(np.random.randint(int(self.class_idx_startend[c][0]), 
                    int(self.class_idx_startend[c][1]),
                    size = 1))

            current_index = (self.batch_index * batch_size) % n # remainder... we randomize anyway so all of this is rather not necessary
            current_batch_size = batch_size
            if n > current_index + batch_size:
                self.batch_index += 1  # batch number within n images
            else:
                self.batch_index = 0 # has gone through all n images, reset batch_index to 0
            self.total_batches_seen += 1
            
            index_array = np.asarray(index_array, dtype=np.int64).flatten()
            yield (index_array, current_index, current_batch_size)

    def next(self):
        '''
        Yields next training batch
        '''
        with self.lock:
            index_array, current_index, current_batch_size = next(self.index_generator)

        # The transformation of images is not under thread lock so it can be done in parallel
        batch_x = np.zeros((current_batch_size,) + self.image_shape, dtype=K.floatx())
        
        # Prepare for random rotations (includes flips)
        if self.image_data_generator.random_rotation:
            batch_flip = np.zeros(current_batch_size)
            batch_rot90 = np.zeros(current_batch_size)

        for i, j in enumerate(index_array):
            fname = self.filenames[j]

            if self.color_mode == "rgb":
                img = coralutils.CoralData(os.path.join(self.directory, fname), load_type="cv2").image[:,:,::-1] # turn BGR into RGB
            else:
                img = coralutils.CoralData(os.path.join(self.directory, fname), load_type="raster").image

            if self.color_mode == "8channel_to_4channel": # assumes target AND source are starting off as 8 channel
                img = np.delete(img, [0,3,5,7], 2) # harded coded to delete channels except for BGR + NIR
            x = img_to_array(img, data_format=self.data_format)

            # apply spectral augmentation if necessary
            if self.image_data_generator.spectral_augmentation:
                x = self.image_data_generator.augmentation_module.apply(x)
            # normalize after spectral shift
            x = self.image_data_generator.generator_normalize(x,
                self.image_data_generator.pixel_mean, 
                self.image_data_generator.pixel_std,
                reverse_normalize = False) 
 
            # random rotations and flips, and save the random # generated
            if self.image_data_generator.random_rotation:
                x, batch_flip[i], batch_rot90[i] = self.image_data_generator.augmentation_module.random_flip_rotation(x, rnd_flip = True, rnd_rotation = True)
            batch_x[i] = x # save edited image

            if self.save_to_dir: # save to directory code
                img_save = self.image_data_generator.generator_normalize(x, 
                    self.image_data_generator.pixel_mean, 
                    self.image_data_generator.pixel_std, 
                    reverse_normalize = True)
                fname = '{prefix}_{index}_{hash}.{format}'.format(prefix = self.save_prefix+'_trainimg',
                                              index = current_index + i,
                                              hash = index_array[i],
                                              format='tif')
                if self.color_mode == "rgb":
                    cv2.imwrite(os.path.join(self.save_to_dir, fname), np.asarray(img_save, dtype=np.uint8)[:,:,::-1])
                else:
                    driver = gdal.GetDriverByName('GTiff')
                    dataset = driver.Create(os.path.join(self.save_to_dir, fname), 
                        img_save.shape[0], 
                        img_save.shape[1], 
                        img_save.shape[2],
                        gdal.GDT_Float32)
                    for chan in range(img_save.shape[2]):
                        dataset.GetRasterBand(chan + 1).WriteArray((img_save[:,:,chan]))
                        dataset.FlushCache()

        # initialize labels
        if self.class_mode == "image":
            batch_y = np.zeros((current_batch_size,) + self.label_shape, dtype = K.floatx()) # N x R x C x D
        elif self.class_mode == "categorical":
            batch_y = np.zeros((current_batch_size,) + self.label_shape, dtype = np.int8) # N x R x C x D
        elif self.class_mode == "zeros":
            batch_y = np.zeros((current_batch_size,) + self.image_size, dtype = K.floatx())
            return batch_x, batch_y
        else: # just return batch_x with no y
            return batch_x

        batch_weights = np.zeros((current_batch_size, self.image_shape[0], self.image_shape[1]), dtype=np.float)
        for i, j in enumerate(index_array):
            label_fname = self.label_filenames[j]

            if self.class_mode == "categorical":
                y = self._load_seg(label_fname)
                if self.image_data_generator.random_rotation:           # flip and rotate according to previous batch_x images
                    y = self.image_data_generator.augmentation_module.flip_rotation(y, batch_flip[i], batch_rot90[i])

                # set weights for image, and put it in batch_weights
                if self.class_weights is not None:
                    weights = np.zeros((self.image_shape[0], self.image_shape[1]), dtype=np.float)
                    for k in self.class_weights:
                        weights[y == self.classes_dict[k]] = self.class_weights[k]             # class_weights must be a dictionary
                    batch_weights[i] = weights

                # save to dir before y is transformed to categorical tensor
                if self.save_to_dir:            
                    labelimg_save = y
                    labelimg_fname = '{prefix}_{index}_{hash}.{format}'.format(prefix = self.save_prefix+'_labelimg',
                                                                          index = current_index + i,
                                                                          hash = index_array[i],
                                                                          format = self.save_format)
                    cv2.imwrite(os.path.join(self.save_to_dir, labelimg_fname), labelimg_save + self.min_labelkey)
                
                y = to_categorical(y, self.num_classes).reshape(self.label_shape) # one hot representation
                batch_y[i] = y

            elif self.class_mode == "image":
                img = coralutils.CoralData(os.path.join(self.label_directory, label_fname), load_type="raster").image # if image and 8channel, then this must also be 8channel
                y = img_to_array(img, data_format=self.data_format)

                if self.image_data_generator.random_rotation:           # flip and rotate according to previous batch_x images
                    y = self.image_data_generator.augmentation_module.flip_rotation(y, batch_flip[i], batch_rot90[i])
                if self.save_to_dir:            # save to dir before y is transformed to categorical tensor
                    labelimg_save = y
                    labelimg_fname = '{prefix}_{index}_{hash}.{format}'.format(prefix = self.save_prefix+'_labelimg',
                                                                          index = current_index + i,
                                                                          hash = index_array[i],
                                                                          format = 'tif')
                    FCNdriver = gdal.GetDriverByName('GTiff')
                    FCNdataset = FCNdriver.Create(os.path.join(self.save_to_dir, labelimg_fname), 
                        labelimg_save.shape[0], 
                        labelimg_save.shape[1], 
                        labelimg_save.shape[2], 
                        gdal.GDT_Float32)
                    for chan in range(labelimg_save.shape[2]):
                        FCNdataset.GetRasterBand(chan+1).WriteArray((labelimg_save[:,:,chan]))
                        FCNdataset.FlushCache()

                y = self.image_data_generator.generator_normalize(y, 
                    self.image_data_generator.pixel_mean, 
                    self.image_data_generator.pixel_std, 
                    reverse_normalize = True)        
                batch_y[i] = y
            
        quick_shape1 = lambda z: np.reshape(z, (z.shape[0],z.shape[1]*z.shape[2],z.shape[3])) # convert to B x (C*R) x N_channel
        quick_shape2 = lambda z: np.reshape(z, (z.shape[0],z.shape[1]*z.shape[2]))

        if not self.reshape:
            return batch_x, batch_y
        else: # formats to temporal so that class_weights can be used
            if self.class_weights is None: 
                return batch_x, quick_shape1(batch_y) 
            else:
                return batch_x, quick_shape1(batch_y), quick_shape2(batch_weights) 

    def _load_seg(self, fn: str) -> np.ndarray:
        """ Load segmented data, which assumes grayscale label image partitioned by 255/num_classes

            fn: filename of the image (with extension suffix
        """
        label_path = os.path.join(self.label_directory, fn)
        if label_path.endswith('.tif'):
            img = pil_image.open(label_path)
            if self.image_size:
                wh_tuple = (self.image_size[1], self.image_size[0])
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

        # Set 0 to 255 gray level to 0 to (num_classes-1)
        for k, v in self.labelkey.items():
            y[y == v] = self.classes_dict[k] - self.min_labelkey
        return y
