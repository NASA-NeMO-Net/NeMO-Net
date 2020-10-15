from typing import Tuple, Callable, List, Union, Dict

import numpy as np
import cv2
import random
import os
import glob
import itertools
import pandas as pd

from osgeo import gdal, ogr, osr
from matplotlib import pyplot as plt
from PIL import Image as pil_image

from NeMO_Utils import apply_channel_corrections, normalize

from keras.preprocessing.image import img_to_array
import keras.backend as K

class CoralData:
	""" Helper class for loading Coral Data 

	# Arguments
		imagepath: Path of image
		labelpath: Path of label image. Expects label image to consist of 0 - num_classes values. Does NOT convert from grayscale (255/num_classes)
			that is stored in training label images for visualization purposes
		bathypath: Path of bathymetry image
		labelkey: key-value pair for class labels. If None, then code will automatically try to infer class labels from .shp files.
			If input is not a .shp file and None is specified, then class_labels will NOT be set (may be appropriate if you only want to load an image
			and class labels are not required)
		load_type: "cv2" for RGB data, or "raster" for satellite multi-channel data
		tfwpath: .tfw file for geotransform data (raster data)
	"""

	def __init__(self, 
		imagepath: str, 
		labelpath: str = None, 
		bathypath: str = None, 
		labelkey: Dict[str, int] = None, 
		load_type: str = "cv2", 
		tfwpath: str = None):

		self.labelimage_consolidated = None
		self.labelimage = None
		self.load_type = load_type
		head, tail = os.path.split(imagepath)
		self.imagefilename = tail

		self.class_labels = None
		self.class_dict = {}
		self.num_classes = 0

		self.image, self.geotransform = self._load_image(imagepath, load_type)

		if load_type == "raster":
			try:
				self.projection = img.GetProjection()
			except Exception:
				self.projection = None
				pass

			if tfwpath is not None:
				tfw_info = np.asarray([float(line.rstrip('\n')) for line in open(tfwpath)]).astype(np.float32)
				# top left x, w-e pixel resolution, 0, top left y, 0, n-s pixel resolution (negative)
				self.geotransform = np.asarray([tfw_info[4], tfw_info[0], tfw_info[1], tfw_info[5], tfw_info[2], tfw_info[3]])

			pixel_size = self.geotransform[1]
			img_xmin = self.geotransform[0]
			img_ymax = self.geotransform[3]

		if labelpath is not None:
			if labelpath.endswith('.shp'): # shape file load
				NoData_value = -1
				class_labels = []
				labelmap = None

				orig_data_source = ogr.Open(labelpath)
				source_ds = ogr.GetDriverByName("Memory").CopyDataSource(orig_data_source, "")
				source_layer = source_ds.GetLayer(0)
				source_srs = source_layer.GetSpatialRef()

				field_vals = list(set([feature.GetFieldAsString('Class_name') for feature in source_layer]))
				field_vals.sort(key=lambda x: x.lower())
				if 'NoData' not in field_vals: 		# NoData field automatically added to .shp files (this is an artifact from PerosBanhos.shp, which is required)
					self.class_labels = ['NoData'] + field_vals 	# all unique labels
					self.class_dict['NoData'] = 0

				x_min, x_max, y_min, y_max = source_layer.GetExtent()

				field_def = ogr.FieldDefn("Class_id", ogr.OFTReal)
				source_layer.CreateField(field_def)
				source_layer_def = source_layer.GetLayerDefn()
				field_index = source_layer_def.GetFieldIndex("Class_id")

				for feature in source_layer:
					val = self.class_labels.index(feature.GetFieldAsString('Class_name'))
					feature.SetField(field_index, val)
					source_layer.SetFeature(feature)
					self.class_dict[feature.GetFieldAsString('Class_name')] = val

				x_res = int((x_max - x_min) / pixel_size)
				y_res = int((y_max - y_min) / pixel_size)
				target_ds = gdal.GetDriverByName('GTiff').Create('temp.tif', x_res, y_res, gdal.GDT_Int32)
				target_ds.SetGeoTransform((x_min, pixel_size, 0, y_max, 0, -pixel_size))
				band = target_ds.GetRasterBand(1)
				band.SetNoDataValue(NoData_value)
				target_ds.SetProjection(self.projection)
				err = gdal.RasterizeLayer(target_ds, [1], source_layer, None, None, [0], options=['ATTRIBUTE=Class_id'])
				if err != 0:
					raise Exception("error rasterizing layer: %s" % err)

				tempband = target_ds.GetRasterBand(1)
				self.labelimage = tempband.ReadAsArray()

				self.num_classes = len(self.class_labels)
				self.labelimage = self.labelimage.astype(np.uint8)
				target_ds = None

				# fix .tif vs .shp dimensional mismatch
				image_xstart = np.max([0, int((x_min - self.geotransform[0])/pixel_size)])
				truth_xstart = np.max([0, int((self.geotransform[0] - x_min)/pixel_size)])
				image_ystart = np.max([0, int((self.geotransform[3] - y_max)/pixel_size)])
				truth_ystart = np.max([0, int((y_max - self.geotransform[3])/pixel_size)])

				total_cols = int((np.min([xsize*pixel_size + self.geotransform[0], x_max]) - np.max([self.geotransform[0], x_min]))/pixel_size)
				total_rows = int((np.min([self.geotransform[3], y_max]) - np.max([-ysize*pixel_size + self.geotransform[3], y_min]))/pixel_size)

				self.image = self.image[image_ystart:image_ystart+total_rows, image_xstart:image_xstart+total_cols, :]
				self.labelimage = self.labelimage[truth_ystart:truth_ystart+total_rows, truth_xstart:truth_xstart+total_cols]
			
			if ((labelpath.endswith('.tif') or labelpath.endswith('.TIF')) or labelpath.endswith('.png')) and load_type is "raster":
				self.labelimage = np.asarray(cv2.imread(labelpath, cv2.IMREAD_UNCHANGED), dtype=np.uint8)

				if self.labelimage is None:
					gdal_truthimg = gdal.Open(labelpath)
					self.labelimage = gdal_truthimg.GetRasterBand(1).ReadAsArray()
					gdal_truthimg_gt = gdal_truthimg.GetGeoTransform()
					if gdal_truthimg_gt[0] == 0 and gdal_truthimg_gt[3] == 0:
						print("labelimage geotransform is not set! Reverting to default image's geotransform...")
						x_min, x_max, y_min, y_max = self.geotransform[0], self.geotransform[0]+self.labelimage.shape[1]*self.geotransform[1], self.geotransform[3]+self.labelimage.shape[0]*self.geotransform[5], self.geotransform[3]
					else:
						x_min, x_max, y_min, y_max = gdal_truthimg_gt[0], gdal_truthimg_gt[0]+self.labelimage.shape[1]*gdal_truthimg_gt[1], gdal_truthimg_gt[3]+self.labelimage.shape[0]*gdal_truthimg_gt[5], gdal_truthimg_gt[3]
					image_xstart = np.max([0, int((x_min - self.geotransform[0])//pixel_size)])
					truth_xstart = np.max([0, int((self.geotransform[0] - x_min)//pixel_size)])
					image_ystart = np.max([0, int((self.geotransform[3] - y_max)//pixel_size)])
					truth_ystart = np.max([0, int((y_max - self.geotransform[3])//pixel_size)])

					total_cols = int((np.min([xsize*pixel_size + self.geotransform[0], x_max]) - np.max([self.geotransform[0], x_min]))//pixel_size)
					total_rows = int((np.min([self.geotransform[3], y_max]) - np.max([-ysize*pixel_size + self.geotransform[3], y_min]))//pixel_size)

					self.image = self.image[image_ystart:image_ystart+total_rows, image_xstart:image_xstart+total_cols, :]
					self.labelimage = self.labelimage[truth_ystart:truth_ystart+int(total_rows*pixel_size//gdal_truthimg_gt[1]), truth_xstart:truth_xstart+int(total_cols*pixel_size//gdal_truthimg_gt[1])]

				gdal_truthimg = None
		if bathypath is not None:
			self.bathyimage = cv2.imread(bathypath, cv2.IMREAD_UNCHANGED)
			self.bathyimage[self.bathyimage == np.min(self.bathyimage)] = -1

		if labelkey is not None: # overwrite class dictionary and labels if provided
			self.class_labels = list(labelkey)
			self.class_dict = labelkey
			self.num_classes = len(self.class_labels) # total number of classes, including those not found

	def consolidate_classes(self, 
		newclassdict: Dict[str, int], 
		transferdict: Dict[str, str]) -> None:
		''' Transfer classes from one dictionary to another (load from a json file)

			newclassdict: new class (final) dict to transfer to
			transferdict: dict that maps old classes to new classes
		'''

		if self.labelimage_consolidated is None:
			self.labelimage_consolidated = np.copy(self.labelimage)
			TF_labelmap = [self.labelimage_consolidated == self.class_dict[k] for k in self.class_dict]
			for counter, k in enumerate(self.class_dict):
				self.labelimage_consolidated[TF_labelmap[counter]] = newclassdict[transferdict[k]]
		else: # consolidated label image already exists
			TF_labelmap = [self.labelimage_consolidated == self.consolidated_class_dict[k] for k in self.consolidated_class_dict]
			for counter, k in enumerate(self.consolidated_class_dict):
				self.labelimage_consolidated[TF_labelmap[counter]] = newclassdict[transferdict[k]]

		self.consolidated_class_dict = newclassdict 		
		self.consolidated_num_classes = len(self.consolidated_class_dict)
		self.consolclass_count = dict((k, (self.labelimage_consolidated == newclassdict[k]).sum()) for k in newclassdict)

	def _load_image(self, 
		imgpath: str,
		loadtype: str = "raster") -> Tuple[np.ndarray, Tuple]:
		''' Load image from imgpath, either raster (using gdal) or RGB (using cv2)

			imgpath: Path to image
			loadtype: "raster" or "rgb"
		'''

		geotransform = None
		if self.load_type == "raster":
			img_gdal = gdal.Open(imgpath)
			geotransform = img_gdal.GetGeoTransform()
			xsize = img_gdal.RasterXSize
			ysize = img_gdal.RasterYSize
			image = np.zeros((ysize, xsize, img_gdal.RasterCount))

			for band in range(img_gdal.RasterCount):
				band += 1
				imgband = img_gdal.GetRasterBand(band)
				image[:,:,band-1] = imgband.ReadAsArray()
		else:
			image = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
			
		return image, geotransform


	def _calculate_corner(self, 
		geotransform: np.ndarray, 
		column: int, 
		row: int) -> Tuple[int, int]:
		''' Calculate new geotransform x,y given amount to offset row and column by
			geotransform: Original geotransform array [top left x, w-e pixel resolution, 0, top left y, 0, n-s pixel resolution (negative)]
			column: number of column pixels to shift by
			row: number of row pixels to shift by
		'''
		x = geotransform[1]*column + geotransform[2]*row + geotransform[0]
		y = geotransform[5]*row + geotransform[4]*column + geotransform[3]
		return x, y

	def _classifyback(self, 
		predictions: np.ndarray) -> np.ndarray: 
		''' Classify from categorical array to label
			# Input:
			# 	predictions: Array of vectorized categorical predictions, nrow x ncol x n_categories
			# Output:
			# 	Array of label predictions, nrow x ncol
		'''
		return np.argmax(predictions,-1)

	def export_consolidated_labelmap(self,
		filename: str) -> None:
		''' Export consolidated labelmap 
			filename: file name to save as
		'''
		driver = gdal.GetDriverByName('GTiff')
		dataset = driver.Create(filename, self.labelimage_consolidated.shape[1], self.labelimage_consolidated.shape[0], 1 , gdal.GDT_Byte)
		if self.geotransform:
			dataset.SetGeoTransform(self.geotransform)
		if self.projection:
			dataset.SetProjection(self.projection)
		outband = dataset.GetRasterBand(1)
		outband.WriteArray(self.labelimage_consolidated[:,:])
		outband.FlushCache()

		outband = None

	def export_raster(self,
		image_array: np.ndarray,
		filepath: str,
		geotransform: Tuple = None,
		projection: str = None,
		bandstoexport: List[int] = None, 
		thresholds: List[float] = None) -> None:
		''' Export image as raster file
			image_array: Array to save
			filepath: filepath and name to save as
			geotransform: geotransform for gdal
			projection: projection for gdal
			bandstoexport: List of channels to export (starting from 0)
			thresholds: min/max thresholds to cutoff at FOR EVERY CHANNEL
		'''

		xsize = image_array.shape[1]
		ysize = image_array.shape[0]
		if bandstoexport:
			channels = len(bandstoexport)
		else:
			channels = image_array.shape[2]
			bandstoexport = [c for c in range(channels)]

		driver = gdal.GetDriverByName('GTiff')
		dataset = driver.Create(filepath, xsize, ysize, channels, gdal.GDT_Float32)

		if geotransform:
			dataset.SetGeoTransform(geotransform)
		if projection:
			dataset.SetProjection(projection)

		for b, chan in enumerate(bandstoexport):
			tempchannel = image_array[:,:,chan]
			if thresholds:
				tempchannel[tempchannel >= thresholds[1]] = thresholds[1]
				tempchannel[tempchannel <= thresholds[0]] = thresholds[0]
			dataset.GetRasterBand(b+1).WriteArray(tempchannel)
			dataset.FlushCache()
		dataset = None

	def export_trainingset(self, 
		exporttrainpath: str, 
		exportlabelpath: str, 
		txtfilename: str, 
		image_size: int = 25, 
		N: int = 100, 
		magnification: float = 1.0, 
		magimg_path: str = None,
		subdir: bool = False, 
		cont: bool = False, 
		consolidated: bool = False, 
		classestoexport: List[str] = None, 
		mosaic_mean: float = 0.0, 
		mosaic_std: float = 1.0, 
		thresholds: List[float] = None,
		bandstoexport: List[int] = None, 
		label_cmap: Tuple = None) -> None:
		''' Export Training/Label dataset from a large mosaic image

			exporttrainpath: Directory for exported patch images
			exportlabelpath: Directory for exported segmented images
			txtfilename: Name of text file to record image characteristics (remember to include '.txt')
			image_size: Size of image (of larger magnified image, if available)
			N: Number of images per class (NOTE: because these are segmented maps, the class is only affiliated with the center pixel)
			magnification: Magnification ratio between label/image (used for super-resolution). Otherwise, 1.
			magimg_path: Directory for magnified image (truth image)
			subdir: Create subdirectories for each class
			cont: Continuously add to folder (True), or overwrite (False)
			consolidated: Export consolidated classes or not
			classestoexport: List of strings of classes to export. 'None' indicates export all classes
			mosaic_mean: Channel means
			mosaic_std: Channel std
			thresholds: [min, max] to threshold output FOR EVERY CHANNEL
			bandstoexport: specific channels to export from raster. None indicates export all channels.
			label_cmap: cmap
		'''

		crop_len = int(np.floor(image_size/2)) # crop_len indicates areas along the borders that we do NOT want to draw from (since we take patches)
		m = magnification
		if bandstoexport is not None:
			num_channels = len(bandstoexport)
		else:
			num_channels = self.image.shape[2]
			bandstoexport = [c for c in range(num_channels)]

		image_mean = apply_channel_corrections(mosaic_mean, self.image.shape[2], 0.0, "mosaic_mean")
		image_std = apply_channel_corrections(mosaic_std, self.image.shape[2], 1.0, "mosaic_std")

		if cont:
			f = open(exporttrainpath + txtfilename,'a')
		else:
			f = open(exporttrainpath + txtfilename,'w')

		# magnified (hi-res) image if path specified
		if magimg_path is not None:
			magimage, _ = self._load_image(magimg_path, self.load_type)

		if consolidated: # make sure that consolidate_classes have been run for this option
			if classestoexport is None:
				export_class_dict = self.consolidated_class_dict 
			else:             
				export_class_dict = {x: self.consolidated_class_dict[x] for x in classestoexport}
			labelimage = self.labelimage_consolidated
			labelcrop = self.labelimage_consolidated[crop_len: self.labelimage_consolidated.shape[0] - crop_len, \
				crop_len: self.labelimage_consolidated.shape[1] - crop_len]
			num_classes = self.consolidated_num_classes
		else:
			if classestoexport is None:
				export_class_dict = self.class_dict
			else:
				export_class_dict = {x:self.class_dict[x] for x in classestoexport}
			labelimage = self.labelimage
			labelcrop = self.labelimage[crop_len: self.labelimage.shape[0] - crop_len, \
				crop_len: self.labelimage.shape[1] - crop_len]
			num_classes = self.num_classes

		# Start export, cycle through all export classes (with or without consolidation)
		classcounter = 0
		for counter, lbl in enumerate(export_class_dict):
			[i,j] = np.where(labelcrop == export_class_dict[lbl]) # find all locations where current class exist
			num_samples = len(i)
			if num_samples != 0: # if there is at least one of this class in the image
				if num_samples < N: # if not enough points to satisfy N
					idx = [count % num_samples for count in range(N)] # repeat same points as required
				else:
					idx = np.asarray(random.sample(range(num_samples), N)).astype(int) # random sample if there are enough points

				subdirpath = '{}/'.format(lbl)
				basecount = 0 # start file index count at 0
				if cont and path.exists(exporttrainpath + subdirpath): # if subdir exists and continuing to add to directory
					basecount = len([f for f in os.listdir(exporttrainpath + subdirpath)]) # count file index from existing files in directory

				for nn, ix in enumerate(idx):
					# Note: i,j are calculated off of labelcrop, and hence they are already centered (only +image_size required to create a patch)
					tempimage = self.image[ int(i[ix]/m): int(i[ix]/m + image_size/m), \
						int(j[ix]/m):int(j[ix]/m + image_size/m), :] # assumes loaded image is the smaller image
					tempimage = normalize(tempimage, image_mean, image_std, False)
					templabelimage = labelimage[i[ix]: i[ix] + image_size, \
						j[ix]: j[ix] + image_size] # assumes large label image
					if magimg_path is not None:
						tempmagimage = magimage[i[ix]: i[ix] + image_size, \
							j[ix]: j[ix] + image_size, :] # assumes large image

					if label_cmap is not None:
						templabel = np.zeros((templabelimage.shape[0], templabelimage.shape[1], 3))
						for count, key in enumerate(export_class_dict):
							templabel[templabelimage == export_class_dict[key]] = np.asarray(label_cmap(count)[-2::-1])*255 # 8bit-based cmap
						templabel = templabel.astype(np.uint8)
					else:
						templabel = np.asarray(templabelimage*(255/num_classes)).astype(np.uint8) # Scale evenly classes from 0-255 in grayscale

					if self.load_type == "raster":
						trainstr = lbl + '_' + str(basecount + nn).zfill(8) + '.tif'
						labelstr = lbl + '_' + str(basecount + nn).zfill(8) + '.tif'
					else:
						trainstr = lbl + '_' + str(basecount + nn).zfill(8) + '.png'
						labelstr = lbl + '_' + str(basecount + nn).zfill(8) + '.png'

					if subdir: # split into subdirectories
						if not os.path.exists(exporttrainpath + subdirpath):
							os.makedirs(exporttrainpath + subdirpath)
						if not os.path.exists(exportlabelpath + subdirpath):
							os.makedirs(exportlabelpath + subdirpath)
						exportimage_filepath = exporttrainpath + subdirpath + trainstr
						exportlabel_filepath = exportlabelpath + subdirpath + labelstr
						# keep track of where we generated this data from
						f.write('./' + subdirpath + trainstr + ' ' + self.imagefilename + ' ' + str(i[ix])+' '+str(j[ix]) + '\n')
					else:
						exportimage_filepath = exporttrainpath + trainstr
						exportlabel_filepath = exportlabelpath + labelstr
						# keep track of where we generated this data from
						f.write('./' + trainstr + ' ' + self.imagefilename + ' ' + str(i[ix])+' '+str(j[ix]) + '\n')

					# save training image
					if self.load_type == "raster":
						x, y = self._calculate_corner(self.geotransform, j[ix] - crop_len, i[ix] - crop_len) # calculate the x,y coordinates
						temp_imagegeotransform = (x, self.geotransform[1], 0, y, 0, self.geotransform[5])
						self.export_raster(tempimage, exportimage_filepath, temp_imagegeotransform, self.projection, bandstoexport, thresholds)
					else:
						tempimage_export = np.zeros((tempimage.shape[0], tempimage.shape[1], num_channels))                  
						for b, chan in enumerate(bandstoexport):
							tempchannel = tempimage[:, :, chan] # Note that no artificial ceiling here, so output might be a float
							if thresholds:
								tempchannel[tempchannel >= thresholds[1]] = thresholds[1]
								tempchannel[tempchannel <= thresholds[0]] = thresholds[0]
							tempimage_export[:, :, b] = tempchannel
						cv2.imwrite(exportimage_filepath, tempimage_export)

					if magimg_path is None: # export regular label image
						cv2.imwrite(exportlabel_filepath, templabel)
					elif self.load_type == "raster": # export magnified image in raster format
						temp_labelgeotransform = (x, self.geotransform[1]*m, 0, y, 0, self.geotransform[5]*m)
						self.export_raster(tempmagimage, exportlabel_filepath, temp_labelgeotransform, self.projection, bandstoexport, thresholds)
					else: # export magnified image in cv2 format
						cv2.imwrite(exportlabel_filepath, tempmagimage)

					print(str(counter*N + nn + 1) + '/ ' + str(len(export_class_dict)*N) +' patches exported', end='\r')
				classcounter += 1
		print("{} of {} total classes found and saved".format(classcounter, len(export_class_dict)))
		f.close()

# #### Load entire line(s) of patches from testimage
# # Input:
# #	image_size: size of image patch
# # 	crop_len: sides to crop (if predicting upon middle pixel)
# # 	offset: offset from top of image
# # 	yoffset: offset from left of image 
# # 	cols: number of columns to output
# # 	lines: Number of lines of patches to output (None defaults to all possible)
# # 	toremove: Remove last channel or not
# # Output:
# # 	whole_dataset: Patch(es) of test image
# 	def _load_whole_data(self, image_size, crop_len, offset=0, yoffset = 0, cols = 1, lines=None, toremove=None):
# 		if image_size%2 == 0:
# 			if lines is None: 	# this is never used currently
# 				lines = self.testimage.shape[0] - 2*crop_len +1

# 			if offset+lines+crop_len > self.testimage.shape[0]+1: # this is never used currently
# 				print("Too many lines specified, reverting to maximum possible")
# 				lines = self.testimage.shape[0] - offset - crop_len

# 			whole_datasets = []
# 			for i in range(offset, lines+offset):
# 				#for j in range(crop_len, self.testimage.shape[1] - crop_len):
# 				for j in range(yoffset, yoffset+cols):
# 					whole_datasets.append(self.testimage[i-crop_len:i+crop_len, j-crop_len:j+crop_len,:])
# 		else:
# 			if lines is None:
# 				lines = self.testimage.shape[0] - 2*crop_len

# 			if offset+lines+crop_len > self.testimage.shape[0]:
# 				print("Too many lines specified, reverting to maximum possible")
# 				lines = self.testimage.shape[0] - offset - crop_len

# 			whole_datasets = []
# 			for i in range(offset, lines+offset):
# 				for j in range(yoffset, yoffset+cols):
# 					whole_datasets.append(self.testimage[i-crop_len:i+crop_len+1, j-crop_len:j+crop_len+1,:])

# 		whole_datasets = np.asarray(whole_datasets)

# 		if toremove is not None:
# 			whole_datasets = np.delete(whole_datasets, toremove, -1)
# 		whole_dataset = self._rescale(whole_datasets)
# 		return whole_dataset

# #### Load entire line(s) of patches from testimage
# # Input:
# #	model: Keras model to predict on
# # 	image_size: size of each image
# # 	num_lines: number of lines to predict
# # 	spacing: Spacing between each image (row,col)
# # 	predict_size: size of prediction area square (starting from center); FCN will use predict_size = image_size
# # 	lastchannelremove: Remove last channel or not
# # 	predict_mid: Only predict the middle of the image (e.g. 256x256 -> only predict on middle 128x128
# # Output:
# # 	whole_predict: Predicted class array
# #   num_predict: Number of times per prediction in array
# # 	prob_predict: Probability of each class per pixel, as calculated by softmax
# # 	truth_predict: Original truth image (cropped)
# # 	accuracy: Overall accuracy of entire prediction
# 	def predict_on_whole_image(self, model, image_size, num_classes, num_lines = None, spacing = (1,1), predict_size = 1, lastchannelremove = True):
# 		crop_len = int(np.floor(image_size/2)) # lengths from sides to not take into account in the calculation of num_lines
# 		offstart = crop_len-int(np.floor(predict_size/2))
            
# 		if image_size%spacing[0] != 0 or image_size%spacing[1] != 0:
# 			print("Error: Spacing does not divide into image size!")
# 			raise ValueError
            
# 		if spacing[0] > predict_size or spacing[1] > predict_size:
# 			print("Spacing must be smaller than predict_size!")
# 			raise ValueError

# 		if image_size%2 == 0:
# 			if num_lines is None:
# 				num_lines = int(np.floor((self.testimage.shape[0] - image_size + spacing[0])/spacing[0])) # Predict on whole image

# 			whole_predict = np.zeros((spacing[0]*(num_lines-1)+predict_size, self.testimage.shape[1]-image_size+predict_size))
# 			num_predict = np.zeros((spacing[0]*(num_lines-1)+predict_size, self.testimage.shape[1]-image_size+predict_size))
# 			prob_predict = np.zeros((spacing[0]*(num_lines-1)+predict_size, self.testimage.shape[1]-image_size+predict_size, num_classes))

# 			truth_predict = self.labelimage[offstart:offstart+whole_predict.shape[0], offstart:offstart+whole_predict.shape[1]]

# 			for offset in range(crop_len, crop_len+spacing[0]*num_lines, spacing[0]):
# 				for cols in range(crop_len, self.testimage.shape[1]-crop_len+1, spacing[1]):
# 					if lastchannelremove:
# 						temp_dataset = self._load_whole_data(image_size, crop_len, offset=offset, yoffset = cols, cols=1, lines=1, toremove=3)
# 					else:
# 						temp_dataset = self._load_whole_data(image_size, crop_len, offset=offset, yoffset = cols, cols=1, lines=1)
# 					# print(temp_dataset.shape)
# 					temp_prob_predict = model.predict_on_batch(temp_dataset)
# 					# print(temp_prob_predict.shape)
# 					temp_predict = self._classifyback(temp_prob_predict)
					
# 					for predict_mat in temp_predict: 	# this is incorrect if temp_predict has more than 1 prediction (e.g. cols>1, lines>1)
# 						whole_predict[offset-crop_len:offset-crop_len+predict_size, cols-crop_len:cols-crop_len+predict_size] = \
# 							whole_predict[offset-crop_len:offset-crop_len+predict_size, cols-crop_len:cols-crop_len+predict_size] + np.reshape(predict_mat, (image_size,image_size))[offstart:offstart+predict_size,offstart:offstart+predict_size]
# 						num_predict[offset-crop_len:offset-crop_len+predict_size, cols-crop_len:cols-crop_len+predict_size] = \
# 							num_predict[offset-crop_len:offset-crop_len+predict_size, cols-crop_len:cols-crop_len+predict_size] + np.ones((image_size,image_size))[offstart:offstart+predict_size,offstart:offstart+predict_size]
# 						prob_predict[offset-crop_len:offset-crop_len+predict_size, cols-crop_len:cols-crop_len+predict_size,:] = \
# 							prob_predict[offset-crop_len:offset-crop_len+predict_size, cols-crop_len:cols-crop_len+predict_size,:] + np.reshape(temp_prob_predict, (1,image_size,image_size,num_classes))[0][offstart:offstart+predict_size,offstart:offstart+predict_size,:]
# # 					print("Line: " + str(offset-crop_len) + " Col: " + str(cols-crop_len) + '/ ' + str(self.testimage.shape[1]-image_size+1) + ' completed')
# 				print("Line: " + str(offset-crop_len) + ' completed')
# 		else:
# 			if num_lines is None:
# 				num_lines = int(np.floor((self.testimage.shape[0] - image_size)/spacing[0])+1) # Predict on whole image

# 			whole_predict = np.zeros((spacing[0]*(num_lines-1)+predict_size, self.testimage.shape[1]-image_size+predict_size))
# 			num_predict = np.zeros((spacing[0]*(num_lines-1)+predict_size, self.testimage.shape[1]-image_size+predict_size))
# 			prob_predict = np.zeros((spacing[0]*(num_lines-1)+predict_size, self.testimage.shape[1]-image_size+predict_size, num_classes))

# 			truth_predict = self.labelimage[offstart:offstart+whole_predict.shape[0], offstart:offstart+whole_predict.shape[1]]

# 			for offset in range(crop_len, crop_len+spacing[0]*num_lines, spacing[0]):
# 				for cols in range(crop_len, self.testimage.shape[1]-crop_len, spacing[1]):
# 					if lastchannelremove:
# 						temp_dataset = self._load_whole_data(image_size, crop_len, offset=offset, yoffset = cols, cols=1, lines=1, toremove=3)
# 					else:
# 						temp_dataset = self._load_whole_data(image_size, crop_len, offset=offset, yoffset = cols, cols=1, lines=1)
# 					temp_predict = model.predict_on_batch(temp_dataset)
# 					temp_predict = self._classifyback(temp_predict)
					
# 					for predict_mat in temp_predict: 	# this is incorrect if temp_predict has more than 1 prediction (e.g. cols>1, lines>1)
# 						whole_predict[offset-crop_len:offset-crop_len+predict_size, cols-crop_len:cols-crop_len+predict_size] = whole_predict[offset-crop_len:offset-crop_len+predict_size, cols-crop_len:cols-crop_len+predict_size] + predict_mat
# 						num_predict[offset-crop_len:offset-crop_len+predict_size, cols-crop_len:cols-crop_len+predict_size] = num_predict[offset-crop_len:offset-crop_len+predict_size, cols-crop_len:cols-crop_len+predict_size] + np.ones(predict_mat.shape)
# # 					print("Line: " + str(offset-crop_len) + " Col: " + str(cols-crop_len) + '/ ' + str(self.testimage.shape[1]-2*crop_len) + ' completed', end=' ')
# 				print("Line: " + str(offset-crop_len) + ' completed', end='/r')

# 		# Remaining code is for the special case in which spacing does not get to the last col/row

# 			# Classify last column of row
# 			# tempcol = whole_predict.shape[1]-image_size
# 			# temp_dataset = self._load_whole_data(image_size, crop_len, offset=offset, yoffset = tempcol, cols=1, lines=1, toremove = 3)
# 			# temp_predict = model.predict_on_batch(temp_dataset)
# 			# temp_predict = self._classifyback(temp_predict)
# 			# whole_predict[offset:offset+image_size, tempcol:tempcol+image_size] = whole_predict[offset:offset+image_size, tempcol:tempcol+image_size] + predict_mat
# 			# num_predict[offset:offset+image_size, tempcol:tempcol+image_size] = num_predict[offset:offset+image_size, tempcol:tempcol+image_size] + np.ones(predict_mat.shape)
		
# 		# Predict on last rows
# 		# if num_lines*spacing[0]+image_size == int(np.floor((self.testimage.shape[0]-image_size)/spacing[0])):
# 		# 	offset = self.testimage.shape[0]-image_size
# 		# 	for cols in range(0, whole_predict.shape[1]-image_size+1, spacing[1]):
# 		# 		if lastchannelremove:
# 		# 			temp_dataset = self._load_whole_data(image_size, crop_len, offset=offset, yoffset = cols, cols=1, lines=1, toremove=3)
# 		# 		else:
# 		# 			temp_dataset = self._load_whole_data(image_size, crop_len, offset=offset, yoffset = cols, cols=1, lines=1)
# 		# 		temp_predict = model.predict_on_batch(temp_dataset)
# 		# 		temp_predict = self._classifyback(temp_predict)
				
# 		# 		for predict_mat in temp_predict:
# 		# 			whole_predict[offset:offset+image_size, cols:cols+image_size] = whole_predict[offset:offset+image_size, cols:cols+image_size] + predict_mat
# 		# 			num_predict[offset:offset+image_size, cols:cols+image_size] = num_predict[offset:offset+image_size, cols:cols+image_size] + np.ones(predict_mat.shape)
# 		# 		print("Line: " + str(offset) + " Col: " + str(cols) + '/ ' + str(whole_predict.shape[1]-image_size+1) + ' completed', end='\r')

# 		# 	# Classify last column of row
# 		# 	tempcol = whole_predict.shape[1]-image_size
# 		# 	temp_dataset = self._load_whole_data(image_size, crop_len, offset=offset, yoffset = tempcol, cols=1, lines=1, toremove = 3)
# 		# 	temp_predict = model.predict_on_batch(temp_dataset)
# 		# 	temp_predict = self._classifyback(temp_predict)
# 		# 	whole_predict[offset:offset+image_size, tempcol:tempcol+image_size] = whole_predict[offset:offset+image_size, tempcol:tempcol+image_size] + predict_mat
# 		# 	num_predict[offset:offset+image_size, tempcol:tempcol+image_size] = num_predict[offset:offset+image_size, tempcol:tempcol+image_size] + np.ones(predict_mat.shape)
		
# 		# whole_predict = np.round(whole_predict.astype(np.float)/num_predict.astype(np.float)).astype(np.uint8)
# 		# class_dict_min = np.min([self.class_dict[k] for k in self.class_dict])
# 		whole_predict = np.round(whole_predict.astype(np.float)/num_predict.astype(np.float))
# 		prob_predict = prob_predict/np.dstack([num_predict.astype(np.float)]*num_classes)
# 		accuracy = 100*np.asarray(whole_predict == truth_predict).astype(np.float32).sum()/(whole_predict.shape[0]*whole_predict.shape[1]) # this is not correct when truth_predict does not start from 0

# 		return whole_predict, num_predict, prob_predict, truth_predict, accuracy

# # Input:
# 	#   cm: confusion matrix from sklearn: confusion_matrix(y_true, y_pred)
# 	#   classes: a list of class labels
# 	#   normalize: whether the cm is shown as number of percentage (normalized)
# 	#   file2sav: unique filename identifier in case of multiple cm
# 	# Output:
# 	#   .png plot of cm
# def plot_confusion_matrix(cm, classes, normalize=False,
# 					title='Confusion matrix',
# 					cmap=plt.cm.Blues, plot_colorbar = False, file2sav = "cm"):
# 	#"""
# 	#This function prints and plots the confusion matrix.
# 	#Normalization can be applied by setting `normalize=True`.
# 	#cm can be a unique file identifier if multiple options exist
# 	#"""
# 	if normalize:
# 		cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
# 		print("Normalized confusion matrix")
# 		tit = "normalized"
# 	else:
# 		print('Confusion matrix, without normalization')
# 		tit = "non_normalized"
# 	# print(cm)

# 	cm_plot = './plots/Confusion_Matrix_' + tit + file2sav + ".png"
# 	plt.figure()
# 	plt.imshow(cm, interpolation='nearest', cmap=cmap)
# 	plt.title(title + " " + tit)
# 	if plot_colorbar:
# 		plt.colorbar(fraction=0.046, pad=0.04)
# 	tick_marks = np.arange(len(classes))
# 	plt.xticks(tick_marks, classes, rotation=90,fontsize=16)
# 	plt.yticks(tick_marks, classes, fontsize=16)

# 	fmt = '.2f' if normalize else 'd'
# 	if normalize:
# 		thresh = 0.5
# 	else:
# 		thresh = cm.max() / 2.
# 	for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
# 		plt.text(j, i, format(cm[i, j], fmt),
# 			horizontalalignment="center", fontsize=16, 
# 			color="white" if cm[i, j] > thresh else "black")
# 		#plt.text(j, i, r'{0:.2f}'.format(cm[i,j]), 
# 	#		horizontalalignment="center", fontsize=16, 
# 	#		color="white" if cm[i, j] > thresh else "black")

# 	#plt.tight_layout()
# 	plt.ylabel('True label', fontsize=16)
# 	plt.xlabel('Predicted label', fontsize=16)
# 	fig = plt.gcf()
# 	fig.set_size_inches(20, 20, forward=True)
# 	plt.show()
# 	# plt.savefig(cm_plot)
# 	# plt.close()


# def confusion_matrix_stats(cm, classes, file2sav = "cm_stats"):
# 	#"""
# 	#This function calculates stats related to the confusion matrix cm
# 	#cm - confusion matrix as numpy array, can be generated or loaded (cm = np.load('./output/cm_whole_image_hyperas1.npy'))
# 	#classes - a list of class labels
# 	#file2save - filename (without csv extension)
# 	#"""
# 	TP = np.trace(cm)
# 	sum_pred = cm.sum(axis=0) # summing over predicted values (each columns going over all rows)
# 	sum_true = cm.sum(axis=1) # summing over true values (each row going over all columns)

# 	total_pred = sum_pred.sum()
# 	total_true = sum_true.sum()

# 	overall_accuracy = (float(TP) / float(total_true))*100.
# 	print("overall_accuracy: " + str(np.round(overall_accuracy, decimals=2)) + "%")

# 	diag = cm.diagonal()
# 	prod_acc = np.true_divide((diag), (sum_true))
# 	user_acc = np.true_divide((diag), (sum_pred))
	
# 	d = {'class_label': label_list, 'prod_acc': prod_acc, 'user_acc': user_acc, 'overall_acc': overall_accuracy/100.}
# 	df = pd.DataFrame(data=d)
	
# 	# df.to_csv('./output/' +file2save + '.csv')

# # fill in last remaining color on a truthmap
# # truthmap_fn: path to truthmap
# # cmap: colormap of colors that already exist in the truthmap
# # lastcolor: last color to fill in, in BGR
# def fill_in_truthmap_lastcolor(truthmap_fn, cmap, lastcolor):
# 	truthmap = cv2.imread(truthmap_fn) 	# Read in as BGR
# 	cmap_8bit = np.asarray([np.asarray(cmap(i)[-2::-1])*255 for i in range(len(cmap.colors))], dtype = np.uint8)	 #in BGR

# 	replacemap = np.ones((truthmap.shape[0], truthmap.shape[1]), dtype=np.bool)
# 	for color in cmap_8bit:
# 		idx = np.where(np.all(truthmap==color, axis=-1))
# 		replacemap[idx] = False
# 	truthmap[replacemap] = lastcolor
# 	return truthmap

# # fill in empty pixels based upon surrounding pixel colors
# # truthmap_fn: path to truthmap (classified from people)
# # surroundingarea: square grid size arround classification pixel (pick an odd number!)
# def fill_in_truthmap(truthmap_fn, surroundingarea, nofillcolor=None):
# 	if nofillcolor is None:
# 		white = np.asarray([255,255,255])
# 	else:
# 		white = nofillcolor
# 	if surroundingarea % 2 == 0:
# 		raise ValueError('Please choose an odd number for fill_in_truthmap surroundingarea')
# 	truthmap = cv2.imread(truthmap_fn)
# 	y,x = np.where(np.all(truthmap == white, axis=-1))
# 	for j,i in zip(y,x):
# 		crop_len = int((surroundingarea-1)/2)
# 		found_replace = False
# 		while found_replace is False:
# 			tempy_min = max(j-crop_len,0)
# 			tempy_max = min(j+crop_len+1,truthmap.shape[0])
# 			tempx_min = max(i-crop_len,0)
# 			tempx_max = min(i+crop_len+1, truthmap.shape[1])
# 			truthmap_patch = truthmap[tempy_min:tempy_max,tempx_min:tempx_max,:]
# 			unq, unq_count = np.unique(truthmap_patch.reshape(-1, truthmap_patch.shape[2]), return_counts=True, axis=0)
# 			idx = np.where(np.all(unq == white, axis=-1))

# 			if len(idx[0]) > 0: 		# Get rid of white counts
# 				unq = np.delete(unq, idx, axis=0)
# 				unq_count = np.delete(unq_count, idx, axis=0)

# 			if len(unq) > 0:			# Make sure there is still at least 1 unique left
# 				found_replace = True
# 			else:						# If no uniques left, increment area by 1
# 				crop_len += 1
# 		maxidx = np.argmax(unq_count)
# 		truthmap[j,i] = unq[maxidx]
# 	return truthmap

# # Loads patch from mosaiced .tif file
# # imagepath: file where raster file is located
# # specific_fn: specific patch filename of the NxN patch being loaded
# # trainfile: file where all info is stored for all NxN patches
# # image_size: Size of image (just 1 number since it is a square)
# # offset: (y,x) offset of patch to load in tuple
# def load_specific_patch(imagepath, specific_fn, trainfile, image_size, offset=0):
# 	f = open(trainfile,"r")
# 	col = []
# 	row = []
# 	rastername = []
# 	patch_name = []
# 	for line in f:
# 		infosplit = line.split(" ")
# 		col.append(int(infosplit[-1]))
# 		row.append(int(infosplit[-2]))
# 		rastername.append(infosplit[-3])
# 		patch_path = ' '.join(infosplit[0:-3])
# 		head, tail = os.path.split(patch_path)
# 		patch_name.append(tail)

# 	idx = patch_name.index(specific_fn)
# 	raster = CoralData(imagepath+'/'+rastername[idx], load_type="raster")
# 	patch = raster.image[row[idx]+offset:row[idx]+offset+image_size, col[idx]+offset:col[idx]+offset+image_size, :]
# 	projection = raster.projection
# 	# geotransform is organized as [top left x, w-e pixel resolution, 0, top left y, 0, n-s pixel resolution (negative)]
# 	# print(raster.geotransform)
# 	geotransform = [g for g in raster.geotransform]
# 	# geotransform = [(g[0]+offset*g[1], g[1], g[2], g[3]+offset*g[5], g[4], g[5]) for g in raster.geotransform]
# 	# geotransform = raster.geotransform
# 	geotransform[0] = geotransform[0] + offset*geotransform[1]
# 	geotransform[3] = geotransform[3] + offset*geotransform[5]

# 	return patch, projection, geotransform

# # Turns RGB cmap into dictionary-defined truthmap
# # Note: Dictionary might go from 1 to num_classes, and hence the grayscale truthmap will go from num_classes:255/num_classes:255
# # RGBpath: path to all RGB truthmaps
# # Graypath: Final path to put all Grayscale truthmaps
# # cmap: RGB cmap
# # class_dict: Dictionary associated with each color to class (make sure it's in same order as cmap!)
# def transform_RGB2Gray(RGBpath, Graypath, cmap, class_dict):
# 	files = [f for f in os.listdir(RGBpath) if os.path.isfile(os.path.join(RGBpath,f))]
# 	classvalues = [class_dict[k] for k in class_dict]
# 	cmap_len = len(cmap.colors)
# 	for f in files:
# 		BGRpatch = cv2.imread(os.path.join(RGBpath,f))
# 		Graypatch = np.zeros((BGRpatch.shape[0],BGRpatch.shape[1]), dtype=np.uint8)
# 		for i in range(cmap_len):
# 			y,x = np.where(np.all(BGRpatch == np.asarray(np.asarray(cmap(i)[-2::-1])*255, dtype=np.uint8), axis=-1))
# 			Graypatch[y,x] = np.uint8(classvalues[i]*255/cmap_len)
# 		cv2.imwrite(os.path.join(Graypath,f), Graypatch)
