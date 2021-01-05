from typing import Tuple, Callable, List, Union, Dict

import numpy as np
import cv2
import random
import os
import glob
import itertools

from osgeo import gdal, ogr, osr
from matplotlib import pyplot as plt
from PIL import Image as pil_image

from NeMO_Utils import apply_channel_corrections, normalize

import keras
from keras.preprocessing.image import img_to_array
from keras.utils.np_utils import to_categorical
from keras.models import load_model
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
		shpfile_classname: The str id for identifying classes within a shapefile (if loading label data from shapefile)
	"""

	def __init__(self, 
		imagepath: str, 
		labelpath: str = None, 
		bathypath: str = None, 
		labelkey: Dict[str, int] = None, 
		load_type: str = "cv2", 
		tfwpath: str = None,
		shpfile_classname: str = 'Class_name'):

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
				num_features = source_layer.GetFeatureCount()

				field_vals = list(set([source_layer[i].GetFieldAsString(shpfile_classname) for i in range(num_features)]))
				field_vals.sort(key=lambda x: x.lower())
				if 'NoData' not in field_vals: 		# NoData field automatically added to .shp files (this is an artifact from PerosBanhos.shp, which is required)
					self.class_labels = ['NoData'] + field_vals 	# all unique labels
					self.class_dict['NoData'] = 0
				print("The following classes were found in the shapefile (NoData is post-added):", self.class_labels)

				field_def = ogr.FieldDefn("Class_id", ogr.OFTReal)
				source_layer.CreateField(field_def)
				source_layer_def = source_layer.GetLayerDefn()
				field_index = source_layer_def.GetFieldIndex("Class_id")

				for feature in source_layer:
					val = labelkey[feature.GetFieldAsString(shpfile_classname)]
					feature.SetField(field_index, val)
					source_layer.SetFeature(feature)
					self.class_dict[feature.GetFieldAsString(shpfile_classname)] = val

				x_min, x_max, y_min, y_max = source_layer.GetExtent()
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
				self.labelimage = self.labelimage.astype(np.uint8)
				# print(self.labelimage)

				self.num_classes = len(self.class_labels)
				target_ds = None

				# fix .tif vs .shp dimensional mismatch
				self.image, self.labelimage = self._fix_image_vs_labels(self.image, 
					self.labelimage, 
					[x_min, x_max], 
					[y_min, y_max], 
					self.geotransform, 
					self.geotransform)
			
			if ((labelpath.endswith('.tif') or labelpath.endswith('.TIF')) or labelpath.endswith('.png')) and load_type is "raster":
				self.labelimage = np.asarray(cv2.imread(labelpath, cv2.IMREAD_UNCHANGED), dtype=np.uint8)

				if self.labelimage is None:
					gdal_labelimg = gdal.Open(labelpath)
					self.labelimage = gdal_labelimg.GetRasterBand(1).ReadAsArray()
					gdal_labelimg_gt = gdal_labelimg.GetGeoTransform()
					if gdal_labelimg_gt[0] == 0 and gdal_labelimg_gt[3] == 0:
						print("labelimage geotransform is not set! Reverting to default image's geotransform...")
						x_min, x_max, y_min, y_max = self.geotransform[0], self.geotransform[0]+self.labelimage.shape[1]*self.geotransform[1], \
							self.geotransform[3]+self.labelimage.shape[0]*self.geotransform[5], self.geotransform[3]
					else:
						x_min, x_max, y_min, y_max = gdal_labelimg_gt[0], gdal_labelimg_gt[0]+self.labelimage.shape[1]*gdal_labelimg_gt[1], \
							gdal_labelimg_gt[3]+self.labelimage.shape[0]*gdal_labelimg_gt[5], gdal_labelimg_gt[3]

					self.image, self.labelimage = self._fix_image_vs_labels(self.image, 
						self.labelimage, 
						[x_min, x_max], 
						[y_min, ymax], 
						self.geotransform, 
						gdal_labelimg_gt)

				gdal_labelimg = None
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

	def _fix_image_vs_labels(self,
		image_array: np.ndarray,
		label_array: np.ndarray,
		xbounds_label: List[float],
		ybounds_label: List[float],
		image_gt: List[float],
		label_gt: List[float]) -> Tuple[np.ndarray, np.ndarray]:
		''' Fixes both image and label arrays so that they are the same size, and correspond to each other pixel-wise

			image_array: multi-channel image
			label_array: label array
			xbounds_label: [min, max] of x coordinates of label, usually taken from geotransform of label image
			ybounds_label: [min, max] of y coordinates of label, usually taken from geotransform of label image
			image_gt: geotransform of image
			label_gt: geotransform of label 
		'''
		x_min, x_max = xbounds_label[0], xbounds_label[1]
		y_min, y_max = ybounds_label[0], ybounds_label[1]
		xsize = image_array.shape[1]
		ysize = image_array.shape[0]
		pixel_size = image_gt[1] # resolution of image

		image_xstart = np.max([0, int((x_min - image_gt[0])//pixel_size)])
		label_xstart = np.max([0, int((image_gt[0] - x_min)//pixel_size)])
		image_ystart = np.max([0, int((image_gt[3] - y_max)//pixel_size)])
		label_ystart = np.max([0, int((y_max - image_gt[3])//pixel_size)])

		total_cols = int((np.min([xsize*pixel_size + image_gt[0], x_max]) - np.max([image_gt[0], x_min]))//pixel_size)
		total_rows = int((np.min([image_gt[3], y_max]) - np.max([-ysize*pixel_size + image_gt[3], y_min]))//pixel_size)

		image = image_array[image_ystart:image_ystart + total_rows, image_xstart:image_xstart + total_cols, :]
		labelimage = label_array[label_ystart:label_ystart + int(total_rows*pixel_size//label_gt[1]), \
			label_xstart:label_xstart + int(total_cols*pixel_size//label_gt[1])]

		return image, labelimage

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

			try:
				self.projection = img_gdal.GetProjection()
			except Exception:
				self.projection = None
				pass
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
