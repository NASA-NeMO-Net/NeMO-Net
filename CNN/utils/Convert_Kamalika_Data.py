import os
import cv2
import numpy as np
from osgeo import gdal, ogr, osr

mat = np.load("train_PerosBanhos_0.npy").item()
trainX = mat['trainX']
trainDX = mat['trainDX']
trainY = mat['trainY']

train_num_start = 0
train_num_end = 1

train_dir = "../Images/Kamalika_Training_Patches/Coral/"
trainref_dir = "../Images/Kamalika_TrainingRef_Patches/Coral/"

num_classes = 10

for i in range(train_num_start, train_num_end):
    trainstr = "Coral_" + str(i).zfill(8) + ".tif"
    img = np.asarray(trainX[i,:,:,:], dtype=np.float32)
    
    driver = gdal.GetDriverByName('GTiff')
    dataset = driver.Create(train_dir + trainstr, img.shape[0], img.shape[1], img.shape[2], gdal.GDT_Float32)
    for chan in range(img.shape[2]):
        dataset.GetRasterBand(chan+1).WriteArray((img[:,:,chan] - 127.5)/127.5)
        dataset.FlushCache()
    
    temptruthimg = trainY[i,:,:]
    templabel = np.asarray(temptruthimg*(255/num_classes)).astype(np.uint8) # Scale evenly classes from 0-255 in grayscale
    cv2.imwrite(trainref_dir+trainstr, templabel)