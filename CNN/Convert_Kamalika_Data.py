import os
import cv2
import numpy as np
from scipy.misc import imresize
from skimage import data, color
from skimage.transform import rescale, resize, downscale_local_mean
from osgeo import gdal, ogr, osr

trainX = np.load("PerosBanhos_training.npy")
trainY = np.load("PerosBanhos_labels.npy")
Sentinel = np.load("PerosBanhos_Sentinel.npy")

train_num_start = 1000
train_num_end = 1500

train_dir = "../Images/Kamalika_Training_Patches/Coral/"
train_LR_dir = "../Images/Kamalika_TrainingLR_Patches/Coral/"
trainref_dir = "../Images/Kamalika_TrainingRef_Patches/Coral/"
train_LRref_dir = "../Images/Kamalika_TrainingLRRef_Patches/Coral/"
valid_dir = "../Images/Kamalika_Valid_Patches/Coral/"
validref_dir = "../Images/Kamalika_ValidRef_Patches/Coral/"
valid_LR_dir = "../Images/Kamalika_ValidLR_Patches/Coral/"
valid_LRref_dir = "../Images/Kamalika_ValidLRRef_Patches/Coral/"
sentinel_dir = "../Images/Kamalika_Sentinel_Patches/Coral/"

num_classes = 10

counter = 0
for i in range(train_num_start, train_num_end):
    trainstr = "Coral_" + str(counter).zfill(8) + ".tif"
    labelstr = "Coral_" + str(counter).zfill(8) + ".png"
    img = np.asarray(trainX[i,:,:,:], dtype=np.float32)
    imgLR = np.zeros((80,80,img.shape[2]))
    for ch in range(0,img.shape[2]):
        imgLR[:,:,ch] = rescale(img[:,:,ch],1.0/5.0,anti_aliasing=True)
    
    driver = gdal.GetDriverByName('GTiff')
    dataset = driver.Create(valid_LR_dir + trainstr, imgLR.shape[0], imgLR.shape[1], imgLR.shape[2], gdal.GDT_Float32)
    for chan in range(imgLR.shape[2]):
        dataset.GetRasterBand(chan+1).WriteArray((imgLR[:,:,chan] - 127.5)/127.5)
        dataset.FlushCache()
    
    
    temptruthimg = trainY[i,:,:]
    temptruthimgLR = np.zeros((80,80))
    for j in range (0,temptruthimgLR.shape[0]):
        for k in range(0,temptruthimgLR.shape[1]):
            temptruthimgLR[j,k] = np.median(temptruthimg[j*5:(j+1)*5,k*5:(k+1)*5])
    temptruthimgLR = np.asarray(temptruthimgLR,dtype=np.uint8)
#     print(np.unique(temptruthimgLR))
    
    templabel = np.asarray(temptruthimgLR*(255/num_classes)).astype(np.uint8) # Scale evenly classes from 0-255 in grayscale
    
    cv2.imwrite(valid_LRref_dir+labelstr, templabel)
    counter = counter+1