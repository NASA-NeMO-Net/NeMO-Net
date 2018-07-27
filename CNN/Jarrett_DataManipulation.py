import sys
sys.path.append("./utils/") # Adds higher directory to python modules path.
import random
import numpy as np
import cv2
import yaml
import glob, os
import shutil
import loadcoraldata_utils as coralutils
import json
import importlib
from NeMO_generator import NeMOImageGenerator, ImageSetLoader
from osgeo import gdal


# Fill in truth maps as taken from Jarrett's submissions
imagepath = "/home/asli/Documents/NeMO-Net Data/"
startpath = "../Images/UserSubmissions/"
finalpath_RGB = "../Images/Jarrett_submissions/Original/"
os.makedirs(os.path.dirname(finalpath_RGB), exist_ok=True)
finalpath_patch = "../Images/Jarrett_submissions/Patches/"
os.makedirs(os.path.dirname(finalpath_patch), exist_ok=True)
finalpath_truthmap = "../Images/Jarrett_submissions/Truthmaps_RGB/"
os.makedirs(os.path.dirname(finalpath_truthmap), exist_ok=True)
rasterfile = startpath + "rastertrain.txt"

image_size = 256
offset = 128

alldir = [os.path.join(startpath,o) for o in os.listdir(startpath) if os.path.isdir(os.path.join(startpath,o))]
counter = 0
for d in alldir:
    files = [f for f in os.listdir(d) if os.path.isfile(os.path.join(d,f))]
    for f in files:
        fillmap = coralutils.fill_in_truthmap(os.path.join(d,f), 3, nofillcolor=np.asarray([0,0,0]))
        filestr = "Coral_" + str(counter).zfill(8) + ".png"
        cv2.imwrite(os.path.join(finalpath_truthmap,filestr), fillmap)
        if f.endswith(".tiff"):
            temp_f = f[:-1] # turn .tiff to .tif
        else:
            temp_f = f
        
        filestr = "Coral_" + str(counter).zfill(8) + ".tif"
        patch, patch_proj, patch_gt = coralutils.load_specific_patch(imagepath, temp_f, rasterfile, image_size, offset=offset)
        patch_R = 255/200*patch[:,:,4]
        patch_G = 255/200*patch[:,:,2]
        patch_B = 255/200*patch[:,:,1]
        patch_R[patch_R > 255] = 255
        patch_G[patch_G > 255] = 255
        patch_B[patch_B > 255] = 255
        patch_BGR = np.rollaxis(np.asarray([patch_B, patch_G, patch_R], dtype=np.uint8),0,3)
        cv2.imwrite(os.path.join(finalpath_RGB,filestr), patch_BGR)
        
        driver = gdal.GetDriverByName('GTiff')
        dataset = driver.Create(finalpath_patch + filestr, image_size, image_size, patch.shape[2], gdal.GDT_Float32)
        for chan in range(patch.shape[2]):
            dataset.GetRasterBand(chan+1).WriteArray((patch[:,:,chan]-100)/100)
            dataset.FlushCache()
        
        counter += 1
        print("Finished output for " + str(counter) + " images")

jsonpath = './utils/CoralClasses.json'
Graypath = '../Images/Jarrett_submissions/Truthmaps_Gray/'
os.makedirs(os.path.dirname(Graypath), exist_ok=True)

with open(jsonpath) as json_file:
    json_data = json.load(json_file)
cmap_Ved = colors.ListedColormap(['xkcd:pink', 'xkcd:grey', 'xkcd:tan', 'xkcd:olive', 'xkcd:forest', 'xkcd:blue',
                                'xkcd:lilac', 'xkcd:cyan', 'xkcd:orange'])
coralutils.transform_RGB2Gray(finalpath_truthmap, Graypath, cmap_Ved, json_data["VedConsolidated_ClassDict"])

# Copy to TrainingPatch_dir
TrainingPatch_dir = '../Images/Jarrett_Training_Patches/Coral/'
os.makedirs(os.path.dirname(TrainingPatch_dir), exist_ok=True)
src_files = os.listdir(finalpath_patch)
for f in src_files:
	full_f = os.path.join(TrainingPatch_dir, f)
	if (os.path.isfile(full_f)):
		shutil.copy(full_f, TrainingPatch_dir)
shutil.copy(rasterfile, '../Images/Jarrett_Training_Patches/')

# Copy to TrainingRef_dir
TrainingRef_dir = '../Images/Jarrett_TrainingRef_Patches/Coral/'
os.makedirs(os.path.dirname(TrainingRef_dir), exist_ok=True)
src_files = os.listdir(Graypath)
for f in src_files:
	full_f = os.path.join(TrainingRef_dir, f)
	if (os.path.isfile(full_f)):
		shutil.copy(full_f, TrainingRef_dir)

# Copy to ValidgPatch_dir
ValidPatch_dir = '../Images/Jarrett_Valid_Patches/Coral/'
os.makedirs(os.path.dirname(ValidPatch_dir), exist_ok=True)
src_files = os.listdir(finalpath_patch)
for f in src_files:
	full_f = os.path.join(ValidPatch_dir, f)
	if (os.path.isfile(full_f)):
		shutil.copy(full_f, ValidPatch_dir)
shutil.copy(rasterfile, '../Images/Jarrett_Valid_Patches/')

# Copy to TrainingRef_dir
ValidRef_dir = '../Images/Jarrett_ValidRef_Patches/Coral/'
os.makedirs(os.path.dirname(ValidRef_dir), exist_ok=True)
src_files = os.listdir(Graypath)
for f in src_files:
	full_f = os.path.join(ValidRef_dir, f)
	if (os.path.isfile(full_f)):
		shutil.copy(full_f, ValidRef_dir)
