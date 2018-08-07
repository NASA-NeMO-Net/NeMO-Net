#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
@author: vjanakir
This code is run first to create the data and folder structure needed for super-resolution code to work.
"""

# load libraries
import os, numpy as np, matplotlib.pyplot as plt
import scipy.io as scio
from osgeo import gdal
from scipy.ndimage import rotate
from scipy.misc import imresize

def rescale(dataset, depth):
    dataset_norm = (dataset.astype(np.float32) - depth/2) / (depth/2)
    return dataset_norm

def normalize_sentinel(image):
    '''
    rescale data to lie between 0 and 255
    '''
    _max = np.max(image)
    _min = np.min(image)
    out = 0 + (image - _min) * (255.-0.) / (_max - _min)
    out = np.array(out, dtype = np.uint8)
    return out

#%% user specified variables

# specify root directory. This is where trainA, trainB, testA, testB etc are created.
rootDir = '/home/vijay/Desktop/coralWork/data/sentinel_wv/'

# specify path to lowres image
lr_file = rootDir + 'Sentinel/STL_4band_2018-02-26.tif'

# specify path to hires image
hr_file = rootDir + 'wv/PerosBanhos_8band.npy'

# specify scaling factor (must be a multiple of 2) that is less than size(high-res image)/size(low-res image)
ratio = 4

# specify size of low-res sub-image to be cropped
lr_imSize = (60, 60)

#%% load lr_image
img = gdal.Open(lr_file)
xsize = img.RasterXSize
ysize = img.RasterYSize
image = np.zeros( (ysize, xsize, img.RasterCount) )

for band in range(img.RasterCount):
    band += 1
    imgband = img.GetRasterBand(band)
    image[:, :, band-1] = imgband.ReadAsArray()
img = image
del image

img[img > 10000] = 10000
img[img < 0] = 0
img = normalize_sentinel(img)
lr_image = img
del img

#%% load hr_image
img = np.load(hr_file)
img = img[:, :, np.array([5, 3, 2, 7]) - 1]
img = normalize_sentinel(img)
hr_image = img
del img

# artificially replace ocean space from zeros to [7,28,35,3]
temp = [7, 28, 35, 3]
for i in range(hr_image.shape[-1]):
    hr_image[:, :, i][hr_image[:, :, i] == 0] = temp[i]

# reshape highres image so that lowres to hires upsampling factor is 4. The actual factor is 5 which cannot be used for superresolution model.
hr_image1 = 0*imresize(hr_image, interp = 'nearest', size = ratio*1./5)
for i in range(hr_image.shape[-1]):
    hr_image1[:,:,i] = imresize(hr_image[:,:,i], interp = 'nearest', size = ratio*1./5)
hr_image = hr_image1
del hr_image1

#%% 
'''
This section of code traverses the low-res and hi-res images, takes a subimage of specified size and does data augmentation. 
The original image as well as augmented images are stored in images_hr and images_lr arrays. The images in the two arrays are aligned.
'''

# augmentation type considered
aug_type = ['none', 'translate', 'rotate90', 'rotate180', 'rotate270', 'flipx', 'flipy', 'missing_channel']

# initialize images_lr and images_hr arrays based on the size of the original images and specified sub-image sizes
xrange_vec_lr = np.arange(0, ( int( np.ceil( lr_image.shape[0] / lr_imSize[0]) ) + 1) * lr_imSize[0], lr_imSize[0])
yrange_vec_lr = np.arange(0, ( int( np.ceil( lr_image.shape[1] / lr_imSize[1]) ) + 1) * lr_imSize[1], lr_imSize[1])
nimages_lr = (len(xrange_vec_lr) - 1) * (len(yrange_vec_lr) - 1)
images_lr = np.nan * np.zeros( (nimages_lr * len(aug_type), lr_imSize[0], lr_imSize[1], lr_image.shape[2]) )

# high-res image size to be cropped - obtained using lowres image size and ratio
hr_imSize = tuple([k*ratio for k in lr_imSize]) 

xrange_vec_hr = np.arange(0, ( int( np.ceil( hr_image.shape[0] / hr_imSize[0]) ) + 1) * hr_imSize[0], hr_imSize[0])
yrange_vec_hr = np.arange(0, ( int( np.ceil( hr_image.shape[1] / hr_imSize[1]) ) + 1) * hr_imSize[1], hr_imSize[1])
nimages_hr = (len(xrange_vec_hr) - 1) * (len(yrange_vec_hr) - 1)
images_hr = np.nan * np.zeros( (nimages_hr * len(aug_type), hr_imSize[0], hr_imSize[1], hr_image.shape[2]) )

if nimages_lr != nimages_hr:
    print("number of lowres images not equal to number of highres images")
    dsfdfg

# do data augmentation and store subimages
imcount = 0
for i in range( len(xrange_vec_lr) - 1 ):
    for j in range( len(yrange_vec_lr) - 1 ):
        for k in range( len(aug_type) ):
            
            if aug_type[k] == 'none': # original image without any transformation
                st_point_lr = (xrange_vec_lr[i], yrange_vec_lr[j])
                st_point_hr = (xrange_vec_hr[i], yrange_vec_hr[j])
                tempimage_lr = lr_image[st_point_lr[0]:st_point_lr[0] + lr_imSize[0], st_point_lr[1]:st_point_lr[1] + lr_imSize[1], :]
                tempimage_hr = hr_image[st_point_hr[0]:st_point_hr[0] + hr_imSize[0], st_point_hr[1]:st_point_hr[1] + hr_imSize[1], :]
            
            elif aug_type[k] == 'translate':
                rand_range=np.random.uniform(-20, 20, (1, 2))
                st_point_lr = (int(max(0, xrange_vec_lr[i] + rand_range[0,0])), int(max(0, yrange_vec_lr[j] + rand_range[0,1])))
                st_point_hr = (int(max(0, xrange_vec_hr[i] + rand_range[0,0] * ratio)), int(max(0, yrange_vec_hr[j] + rand_range[0,1] * ratio)))
                tempimage_lr = lr_image[st_point_lr[0]:st_point_lr[0] + lr_imSize[0], st_point_lr[1]:st_point_lr[1] + lr_imSize[1], :]
                tempimage_hr = hr_image[st_point_hr[0]:st_point_hr[0] + hr_imSize[0], st_point_hr[1]:st_point_hr[1] + hr_imSize[1], :]
            
            elif aug_type[k] == 'rotate90':
                st_point_lr = (xrange_vec_lr[i], yrange_vec_lr[j])
                st_point_hr = (xrange_vec_hr[i], yrange_vec_hr[j])
                tempimage_lr = lr_image[st_point_lr[0]:st_point_lr[0] + lr_imSize[0], st_point_lr[1]:st_point_lr[1] + lr_imSize[1], :]
                tempimage_lr = rotate(tempimage_lr, 90)
                tempimage_hr = hr_image[st_point_hr[0]:st_point_hr[0] + hr_imSize[0], st_point_hr[1]:st_point_hr[1] + hr_imSize[1], :]
                tempimage_hr = rotate(tempimage_hr, 90)
            
            elif aug_type[k] == 'rotate180':
                st_point_lr = (xrange_vec_lr[i], yrange_vec_lr[j])
                st_point_hr = (xrange_vec_hr[i], yrange_vec_hr[j])
                tempimage_lr = lr_image[st_point_lr[0]:st_point_lr[0] + lr_imSize[0], st_point_lr[1]:st_point_lr[1] + lr_imSize[1], :]
                tempimage_lr = rotate(tempimage_lr, 180)
                tempimage_hr = hr_image[st_point_hr[0]:st_point_hr[0] + hr_imSize[0], st_point_hr[1]:st_point_hr[1] + hr_imSize[1], :]
                tempimage_hr = rotate(tempimage_hr, 180)
            
            elif aug_type[k] == 'rotate270':
                st_point_lr = (xrange_vec_lr[i], yrange_vec_lr[j])
                st_point_hr = (xrange_vec_hr[i], yrange_vec_hr[j])
                tempimage_lr = lr_image[st_point_lr[0]:st_point_lr[0] + lr_imSize[0], st_point_lr[1]:st_point_lr[1] + lr_imSize[1], :]
                tempimage_lr = rotate(tempimage_lr, 270)
                tempimage_hr = hr_image[st_point_hr[0]:st_point_hr[0] + hr_imSize[0], st_point_hr[1]:st_point_hr[1] + hr_imSize[1], :]
                tempimage_hr = rotate(tempimage_hr, 270)
            
            elif aug_type[k] == 'flipx':
                st_point_lr = (xrange_vec_lr[i], yrange_vec_lr[j])
                st_point_hr = (xrange_vec_hr[i], yrange_vec_hr[j])
                tempimage_lr = lr_image[st_point_lr[0]:st_point_lr[0] + lr_imSize[0], st_point_lr[1]:st_point_lr[1] + lr_imSize[1], :]
                tempimage_lr = np.fliplr(tempimage_lr)
                tempimage_hr = hr_image[st_point_hr[0]:st_point_hr[0] + hr_imSize[0], st_point_hr[1]:st_point_hr[1] + hr_imSize[1], :]
                tempimage_hr = np.fliplr(tempimage_hr)
            
            elif aug_type[k] == 'flipy':
                st_point_lr = (xrange_vec_lr[i], yrange_vec_lr[j])
                st_point_hr = (xrange_vec_hr[i], yrange_vec_hr[j])
                tempimage_lr = lr_image[st_point_lr[0]:st_point_lr[0] + lr_imSize[0], st_point_lr[1]:st_point_lr[1] + lr_imSize[1], :]
                tempimage_lr = np.flipud(tempimage_lr)
                tempimage_hr = hr_image[st_point_hr[0]:st_point_hr[0] + hr_imSize[0], st_point_hr[1]:st_point_hr[1] + hr_imSize[1], :]
                tempimage_hr = np.flipud(tempimage_hr)
            
            elif aug_type[k] == 'missing_channel': # a random channel is made zero
                st_point_lr = (xrange_vec_lr[i], yrange_vec_lr[j])
                st_point_hr = (xrange_vec_hr[i], yrange_vec_hr[j])
                elimidx = np.random.choice([0, 1, 2])
                tempimage_lr = lr_image[st_point_lr[0]:st_point_lr[0] + lr_imSize[0], st_point_lr[1]:st_point_lr[1] + lr_imSize[1], :]
                tempimage_lr[:, :, elimidx] = 0
                tempimage_hr = hr_image[st_point_hr[0]:st_point_hr[0] + hr_imSize[0], st_point_hr[1]:st_point_hr[1] + hr_imSize[1], :]
                tempimage_hr[:, :, elimidx] = 0
            
            # proportion of missing data (zeros) in subimage less than 30%
            missing_data_condition = ( np.where(tempimage_hr == 0)[0].shape[0]*1. / np.prod(tempimage_hr.shape) ) < 0.3
            
            # low-res image width and height same
            lowres_size_match = min(tempimage_lr[:, :, 0].shape) == max(tempimage_lr[:, :, 0].shape)
            
            # high-res image width and height same
            highres_size_match = min(tempimage_hr[:, :, 0].shape) == max(tempimage_hr[:, :, 0].shape)
            
            # if the below condition is true, save that sub image, else discard
            if ( (lowres_size_match) & (highres_size_match) & (missing_data_condition) ):
                images_lr[imcount, :tempimage_lr.shape[0], :tempimage_lr.shape[1], :] = tempimage_lr
                images_hr[imcount, :tempimage_hr.shape[0], :tempimage_hr.shape[1], :] = tempimage_hr
                imcount = imcount + 1
            else:
                pass

images_lr = images_lr[:imcount, :, :, :]
images_hr = images_hr[:imcount, :, :, :]
images_lr = np.array(images_lr, dtype = np.uint8)
images_hr = np.array(images_hr, dtype = np.uint8)

#%% visuallycheck if aligned images in low-res ang high-res

for k in range(imcount):
    plt.figure(k+1)
    
    plt.subplot(2,1,1)
    plt.imshow(images_lr[k,:,:,:3])
    
    plt.subplot(2,1,2)
    plt.imshow(images_hr[k,:,:,:3])
    
    plt.show()
    
    input("enter...")
    
    plt.close("all")
    
#%% make directory structure 
'''
This section of code takes the low-res and high-res sub-image arrays and saves data in the right folders (train, test, valid).
'''

# make root directory as current
os.chdir(rootDir)

# create the below folders
os.system("mkdir trainA trainB validA validB testA testB vis_images save_models")
os.system("mkdir dataA dataB")

# save high-res images
for i in range(images_hr.shape[0]):
    savedict = {}
    savedict['data'] = images_hr[i]
    scio.savemat(rootDir + 'dataA/' + str(i) + '.mat', savedict)

filelist = os.listdir(rootDir + 'dataA/')
filelist = [k for k in filelist if '.mat' in k]

# get file indices
idx = np.arange( len(filelist) )

# shuffle the indices randomly
np.random.shuffle(idx)

# train folder has 60% data
ntrain = int(0.6 * idx.shape[0])

# valid and test folders have 20% each
nvalid = int(0.8 * idx.shape[0])

# move the sub-images to train, valid and test folders
os.chdir(rootDir + 'dataA/')
os.system("mv " + " ".join([ filelist[k] for k in np.arange( len(filelist) ) if k in idx[:ntrain] ]) + " ../trainA/")
os.system("mv " + " ".join([ filelist[k] for k in np.arange( len(filelist) ) if k in idx[ntrain:nvalid] ]) + " ../validA/")
os.system("mv " + " ".join([ filelist[k] for k in np.arange( len(filelist) ) if k in idx[nvalid:] ]) + " ../testA/")

# The matching files of low-res are saved in folders with subscript "B"
for i in range( images_lr.shape[0] ):
    savedict = {}
    savedict['data'] = images_lr[i]
    scio.savemat(rootDir + 'dataB/' + str(i) + '.mat', savedict)

filelist = os.listdir(rootDir + 'dataB/')
filelist = [k for k in filelist if '.mat' in k]
os.chdir(rootDir + 'dataB/')
os.system("mv " + " ".join([ filelist[k] for k in np.arange( len(filelist) ) if k in idx[:ntrain] ]) + " ../trainB/")
os.system("mv " + " ".join([ filelist[k] for k in np.arange( len(filelist) ) if k in idx[ntrain:nvalid] ]) + " ../validB/")
os.system("mv " + " ".join([ filelist[k] for k in np.arange( len(filelist) ) if k in idx[nvalid:] ]) + " ../testB/")