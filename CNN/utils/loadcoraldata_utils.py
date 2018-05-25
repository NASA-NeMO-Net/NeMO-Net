import numpy as np
import cv2
import random
import os
import glob
from osgeo import gdal, ogr, osr
from matplotlib import pyplot as plt
from PIL import Image as pil_image
from keras.preprocessing.image import img_to_array
from sklearn.metrics import confusion_matrix
from pandas_ml import ConfusionMatrix
import keras.backend as K
import itertools
import matplotlib.pyplot as plt
import pandas as pd


# Class of coral data, consisting of an image and possibly a corresponding truth map
class CoralData:
	image = None
	num_classes = 0
	truthimage = None
	testimage = None
	train_datasets = None
	train_labels = None
	valid_datasets = None
	valid_labels = None
	test_datasets = None
	test_labels = None
	load_type = None
	projection = None
	depth = 255

	def __init__(self, Imagepath, Truthpath=None, Testpath = None, Bathypath=None, truth_key=None, load_type="cv2", tfwpath=None):
		# Load images
		self.truthimage_consolidated = None
		self.load_type = load_type
		head, tail = os.path.split(Imagepath)
		self.imagefilename = tail
		if load_type == "PIL":
			self.image = img_to_array(pil_image.open(Imagepath))
			if Truthpath is not None:
				self.truthimage = cv2.imread(Truthpath, cv2.IMREAD_UNCHANGED)
			# if Testpath is not None:
			# 	self.testimage = img_to_array(pil_image.open(Testpath))
		elif load_type == "cv2":
			self.image = cv2.imread(Imagepath, cv2.IMREAD_UNCHANGED)
			if Truthpath is not None:
				self.truthimage = cv2.imread(Truthpath, cv2.IMREAD_UNCHANGED)
			# if Testpath is not None:
			# 	self.testimage = cv2.imread(Testpath, cv2.IMREAD_UNCHANGED)
		elif load_type == "raster":
			img = gdal.Open(Imagepath)
			try:
				self.projection = img.GetProjection()
			except Exception:
				pass
			xsize = img.RasterXSize
			ysize = img.RasterYSize

			if tfwpath is not None:
				tfw_info = np.asarray([float(line.rstrip('\n')) for line in open(tfwpath)]).astype(np.float32)
				# top left x, w-e pixel resolution, 0, top left y, 0, n-s pixel resolution (negative)
				self.geotransform = np.asarray([tfw_info[4], tfw_info[0], tfw_info[1], tfw_info[5], tfw_info[2], tfw_info[3]])
			else:
				self.geotransform = img.GetGeoTransform()
			pixel_size = self.geotransform[1]
			img_xmin = self.geotransform[0]
			img_ymax = self.geotransform[3]

			self.image = np.zeros((ysize,xsize,img.RasterCount))

			for band in range(img.RasterCount):
				band += 1
				imgband = img.GetRasterBand(band)
				self.image[:,:,band-1] = imgband.ReadAsArray()
			img = None
		else:
			print("Load type error: specify either PIL, cv2, or raster")
			return None

		if Truthpath is not None:
			if Truthpath.endswith('.shp'):
				NoData_value = -1
				class_labels = []
				labelmap = None

				orig_data_source = ogr.Open(Truthpath)
				source_ds = ogr.GetDriverByName("Memory").CopyDataSource(orig_data_source, "")
				source_layer = source_ds.GetLayer(0)
				source_srs = source_layer.GetSpatialRef()

				field_vals = list(set([feature.GetFieldAsString('Class_name') for feature in source_layer]))
				field_vals.sort(key=lambda x: x.lower())
				if 'NoData' not in field_vals:
					self.class_labels = ['NoData'] + field_vals 	# all unique labels

				x_min, x_max, y_min, y_max = source_layer.GetExtent()

				field_def = ogr.FieldDefn("Class_id", ogr.OFTReal)
				source_layer.CreateField(field_def)
				source_layer_def = source_layer.GetLayerDefn()
				field_index = source_layer_def.GetFieldIndex("Class_id")

				for feature in source_layer:
					val = self.class_labels.index(feature.GetFieldAsString('Class_name'))
					feature.SetField(field_index, val)
					source_layer.SetFeature(feature)

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
				self.truthimage = tempband.ReadAsArray()

				num_classes = len(self.class_labels)
				# print("truth classes: ", truth_classes)
				# for c in truth_classes:
				# 	print('Class {c} contains {n} pixels'.format(c=c, n=(self.truthimage == c).sum()))
				self.truthimage = self.truthimage.astype(np.uint8)
				target_ds = None

				# fix .tif vs .shp dimensional mismatch
				image_xstart = np.max([0, int((x_min - self.geotransform[0])/pixel_size)])
				truth_xstart = np.max([0, int((self.geotransform[0] - x_min)/pixel_size)])
				image_ystart = np.max([0, int((self.geotransform[3] - y_max)/pixel_size)])
				truth_ystart = np.max([0, int((y_max - self.geotransform[3])/pixel_size)])

				total_cols = int((np.min([xsize*pixel_size + self.geotransform[0], x_max]) - np.max([self.geotransform[0], x_min]))/pixel_size)
				total_rows = int((np.min([self.geotransform[3], y_max]) - np.max([-ysize*pixel_size + self.geotransform[3], y_min]))/pixel_size)

				self.image = self.image[image_ystart:image_ystart+total_rows, image_xstart:image_xstart+total_cols, :]
				self.truthimage = self.truthimage[truth_ystart:truth_ystart+total_rows, truth_xstart:truth_xstart+total_cols]
				self.class_weights = dict((i,(self.truthimage.shape[0]*self.truthimage.shape[1])/(self.truthimage==i).sum()) for i in range(num_classes))
			if ((Truthpath.endswith('.tif') or Truthpath.endswith('.TIF')) or Truthpath.endswith('.png')) and load_type is "raster":
				self.truthimage = cv2.imread(Truthpath)
				class_indices = np.unique(self.truthimage)
				num_classes = len(class_indices)
				try:
					self.class_weights = dict((i,(self.truthimage.shape[0]*self.truthimage.shape[1])/(self.truthimage==i).sum()) for i in class_indices)

					gdal_truthimg = gdal.Open(Truthpath)
					gdal_truthimg_gt = gdal_truthimg.GetGeoTransform()
					x_min, x_max, y_min, y_max = gdal_truthimg_gt[0], gdal_truthimg_gt[0]+self.truthimage.shape[1]*gdal_truthimg_gt[1], \
						gdal_truthimg_gt[3]-self.truthimage.shape[0]*gdal_truthimg_gt[1], gdal_truthimg_gt[3]
					
					image_xstart = np.max([0, int((x_min - self.geotransform[0])/pixel_size)])
					truth_xstart = np.max([0, int((self.geotransform[0] - x_min)/pixel_size)])
					image_ystart = np.max([0, int((self.geotransform[3] - y_max)/pixel_size)])
					truth_ystart = np.max([0, int((y_max - self.geotransform[3])/pixel_size)])

					total_cols = int((np.min([xsize*pixel_size + self.geotransform[0], x_max]) - np.max([self.geotransform[0], x_min]))/pixel_size)
					total_rows = int((np.min([self.geotransform[3], y_max]) - np.max([-ysize*pixel_size + self.geotransform[3], y_min]))/pixel_size)

					self.image = self.image[image_ystart:image_ystart+total_rows, image_xstart:image_xstart+total_cols, :]
					self.truthimage = self.truthimage[truth_ystart:truth_ystart+total_rows, truth_xstart:truth_xstart+total_cols]
				except:
					print("Warning! Truth image not in expected format... loading directly whole image using cv2...")

				gdal_truthimg = None
		if Bathypath is not None:
			self.bathyimage = cv2.imread(Bathypath, cv2.IMREAD_UNCHANGED)
			self.bathyimage[self.bathyimage == np.min(self.bathyimage)] = -1

	    # Set labels from 0 to item_counter based upon input truth_key
		if truth_key is not None:
			item_counter = 0
			if isinstance(truth_key,(dict,)):  
				self.class_labels = list(truth_key)
				self.class_dict = truth_key
				self.num_classes = len(self.class_labels) # total number of classes, including those not found
			else:
				for item in truth_key: # This was explicitly designed for 4 class Samoa data that comes in different shades of gray, ignore for now
					self.truthimage[self.truthimage == item] = item_counter
					item_counter+=1
				self.num_classes = len(np.unique(self.truthimage))
		else:
			try:
				self.num_classes = len(self.class_labels)
				self.class_dict = dict((self.class_labels[i],i) for i in range(self.num_classes))
			except:
				pass

#### Transfer classes from one dictionary to another (load from a json file)
# Input:
# 	newclassdict: new class (final) dict to transfer to
# 	transferdict: dict that maps old classes to new classes
	def Consolidate_classes(self, newclassdict, transferdict):
		if self.truthimage_consolidated is None:
			self.truthimage_consolidated = np.copy(self.truthimage)
			TF_labelmap = [self.truthimage_consolidated == self.class_dict[k] for k in self.class_dict]
			counter = 0
			for k in self.class_dict:
				self.truthimage_consolidated[TF_labelmap[counter]] = newclassdict[transferdict[k]]
				counter += 1
		else:
			TF_labelmap = [self.truthimage_consolidated == self.consolidated_class_dict[k] for k in self.consolidated_class_dict]
			counter = 0
			for k in self.consolidated_class_dict:
				self.truthimage_consolidated[TF_labelmap[counter]] = newclassdict[transferdict[k]]
				counter += 1

		self.consolidated_class_dict = newclassdict 		
		self.consolidated_num_classes = len(self.consolidated_class_dict)
		# Need to worry about divide by zero error
		self.consolclass_weights = dict((k, (self.truthimage_consolidated.shape[0]*self.truthimage_consolidated.shape[1])/(self.truthimage_consolidated==newclassdict[k]).sum()) for k in newclassdict)
		for k in self.consolclass_weights:
			if self.consolclass_weights[k] == float("inf"):
				self.consolclass_weights[k] = 0
		self.consolclass_count = dict((k, (self.truthimage_consolidated == newclassdict[k]).sum()) for k in newclassdict)

	def export_consolidated_truthmap(self, filename):
		driver = gdal.GetDriverByName('GTiff')
		# print(self.image.shape)
		dataset = driver.Create(filename, self.image.shape[1], self.image.shape[0], 1 , gdal.GDT_Byte)
		dataset.SetGeoTransform(self.geotransform)
		dataset.SetProjection(self.projection)
		# print(self.truthimage_consolidated.shape)
		outband = dataset.GetRasterBand(1)
		outband.WriteArray(self.truthimage_consolidated[:,:])
		outband.FlushCache()
		# dataset.GetRasterBand(1).WriteArray(self.truthimage_consolidated)
		# dataset.FlushCache()

	def get_anchors(self, classdict, anchorlist):
		channels = self.image.shape[2]
		if self.truthimage_consolidated is None:
			TF_labelmap = [self.truthimage== classdict[k] for k in anchorlist]
		else:
			TF_labelmap = [self.truthimage_consolidated == classdict[k] for k in anchorlist]

		# anchormean = np.asarray([np.mean(self.image[TF_labelmap[i]], axis=0) for i in range(len(anchorlist))]) 		# This erroneously includes all the zero points of the data
		anchormin = []
		anchormean = []
		anchorstd = []
		# all values calculated inside the for loop will NOT include 0 values
		for i in range(len(anchorlist)):
			transpose_minmap = np.transpose(self.image[TF_labelmap[i]])
			tempmin = np.squeeze(np.asarray([np.min(transpose_minmap[j][np.nonzero(transpose_minmap[j])]) for j in range(channels)]))
			tempmean = np.squeeze(np.asarray([np.mean(transpose_minmap[j][np.nonzero(transpose_minmap[j])]) for j in range(channels)]))
			tempstd = np.squeeze(np.asarray([np.std(transpose_minmap[j][np.nonzero(transpose_minmap[j])]) for j in range(channels)]))
			anchormin.append(tempmin)
			anchormean.append(tempmean)
			anchorstd.append(tempstd)

		anchormin = np.asarray(anchormin)
		anchormean = np.asarray(anchormean)
		anchorstd = np.asarray(anchorstd)
		anchormax = np.asarray([np.max(self.image[TF_labelmap[i]], axis=0) for i in range(len(anchorlist))])
		return anchormean, anchorstd, anchormin, anchormax

	def load_PB_consolidated_classes(self):
		self.PB_LOF2consolclass = {"NoData": "Other", "Clouds": "Other", "deep lagoonal water": "Other", "deep ocean water": "Other", "Inland waters": "Other", 
			"mangroves": "Other", "terrestrial vegetation": "Other", "Wetlands": "Other",
			"back reef - pavement": "Pavement with algae",
			"back reef - rubble dominated": "Rubble with CCA",
			"back reef - sediment dominated": "Sediment bare", "Beach": "Sediment bare", "fore reef sand flats": "Sediment bare", 
			"lagoonal sediment apron - sediment dominated": "Sediment bare", "lagoonal floor - barren": "Sediment bare", 
			"dense seagrass meadows": "Seagrass",
			"Rocky beach": "Rock",
			"coralline algal ridge (reef crest)": "Reef crest",
			"deep fore reef slope": "Low Relief HB", "shallow fore reef slope": "Low Relief HB", "shallow fore reef terrace": "Low Relief HB",
			"back reef coral framework": "High Relief HB", "lagoonal fringing reefs": "High Relief HB",
			"lagoonal patch reefs": "Patch reefs"
			}
		self.PB_consolidated_classes = {"Other": 0, "Pavement with algae": 1, "Rubble with CCA": 2, "Sediment bare": 3, "Seagrass": 4, "Rock": 5,
			"Reef crest": 6, "Low Relief HB": 7, "High Relief HB": 8, "Patch reefs": 9}

		for k in self.class_labels:
			self.consol_labels = dict((k,self.PB_consolidated_classes[self.PB_LOF2consolclass[k]]) for k in self.class_labels)

		self.truthimage_consolidated = np.copy(self.truthimage)
		#This might have errors when reassigning labels!!!
		for i in range(len(self.class_labels)):
			self.truthimage_consolidated[self.truthimage_consolidated == i] = self.PB_consolidated_classes[self.PB_LOF2consolclass[self.class_labels[i]]]
		self.consolclass_weights = dict((i, (self.truthimage_consolidated.shape[0]*self.truthimage_consolidated.shape[1])/(self.truthimage_consolidated==i).sum()) for i in range(len(self.PB_consolidated_classes)))

#### Load Image
# Input:
# 	Imagepath: Path to Image
	def load_image(self, Imagepath, load_type="cv2"):
		if load_type == "PIL":
			self.image = pil_image.open(img_to_array(Imagepath))
		elif load_type == "cv2":
			self.image = cv2.imread(Imagepath,cv2.IMREAD_UNCHANGED)
		else:
			print("Load type error: specify either PIL, cv2, or raster")

#### Load Reference/Truth Image
# Input:
# 	Truthpath: Path to Reference Image
# 	truth_key: Convert key to 0:len(truth_key)
# 		For our purposes, 0=Sand, 1=Branching, 2=Mounding, 3=Rock
	def load_truthimage(self, Truthpath, truth_key=None):
		self.truthimage = cv2.imread(Truthpath, cv2.IMREAD_UNCHANGED)
		# Set labels from 0 to item_counter based upon input truth_key
		if truth_key is not None:
			item_counter = 0
			for item in truth_key:
				self.truthimage[self.truthimage == item ] = item_counter 
				item_counter+=1
			self.num_classes = len(np.unique(self.truthimage))

#### Set the pixel depth (usually 8-bit = 255)
# Input:
# 	depth: Pixel depth
	def set_depth(self, depth):
		self.depth = depth


	def _calculate_corner(self, geotransform, column, row):
		x = geotransform[1]*column + geotransform[2]*row + geotransform[0]
		y = geotransform[5]*row + geotransform[4]*column + geotransform[3]
		return x, y
#### Normalize Image
# Input:
# 	dataset: set of vectorized images, N_images x nrow x ncol x n_channels
# Output:
# 	Set of vectorized normalized images, N_images x nrow x ncol x n_channels
	def _rescale(self, dataset):
		return (dataset.astype(np.float32) - self.depth/2)/(self.depth/2)

#### Classify from categorical array to label
# Input:
# 	predictions: Array of vectorized categorical predictions, nrow x ncol x n_categories
# Output:
# 	Array of label predictions, nrow x ncol
	def _classifyback(self, predictions):
		return np.argmax(predictions,-1)

#### Randomize a set of images
# Input:
# 	dataset: set of images, N_images x nrow x ncol x n_channels
# 	labels: set of labels, N_images x num_labels
# Output:
# 	shuffled_dataset: set of randomized images, N_images x nrow x ncol x n_channels
# 	shuffled_labels: set of randomized labels, N_images x num_labels
	def _randomize(self, dataset, labels):
	    permutation = np.random.permutation(labels.shape[0])
	    shuffled_dataset = dataset[permutation,:,:,:]
	    shuffled_labels = labels[permutation,:]
	    return shuffled_dataset, shuffled_labels 

#### Generate a randomized, normalized set of points/labels from original image
# Input:
# 	N: Number of points per class
#	consolidated: consolidated classes
#	bandstoexport: specific bands to export (starting at 1)
# Output:
# 	points: set of randomized points, n x n_channels (defined by bandstoexport) 
# 	labels: set of randomized labels (concurrent with points), n
	def generate_randomized_points(self, N, consolidated = False, bandstoexport=None, cmap = None):

		if consolidated:
			export_class_dict = self.consolidated_class_dict
			truthimage = self.truthimage_consolidated
			num_classes = self.consolidated_num_classes
		else:
			export_class_dict = self.class_dict
			truthimage = self.truthimage
			num_classes = self.num_classes

		points = []
		labels = []

		for k in export_class_dict:
			if cmap is None:
				[i,j] = np.where(truthimage == export_class_dict[k])
			else:
				# truthimage is in BGR here
				[i,j] = np.where(np.all(truthimage == np.asarray(np.asarray(cmap(export_class_dict[k]-1)[-2::-1])*255, dtype=np.uint8), axis=-1))
			if len(i) != 0:
				if len(i) < N:
					idx = [count%len(i) for count in range(N)]
				else:
					idx = np.asarray(random.sample(range(len(i)), N)).astype(int)

				for nn in range(len(idx)):
					if bandstoexport is not None:
						bandstoexport = np.asarray(bandstoexport)
						points.append(self.image[i[idx[nn]],j[idx[nn]],bandstoexport-1])
					else:
						points.append(self.image[i[idx[nn]],j[idx[nn]],:])
					labels.append(truthimage[i[idx[nn]], j[idx[nn]]])

		points = np.asarray(points)
		labels = np.asarray(labels)

		return points, labels

# Input:
#	exporttrainpath: Directory for exported patch images 
# 	exportlabelpath: Directory for exported segmented images
# 	txtfilename: Name of text file to record image names (remember to include '.txt')
# 	image_size: Size of image
# 	N: Number of images per class (NOTE: because these are segmented maps, the class is only affiliated with the center pixel)
# 	lastchannelremove: Remove last channel or not
# 	labelkey: Naming convention of class labels (NOTE: must be same # as the # of classes)
# 	subdir: Create subdirectories for each class
# 	cont: Continuously add to folder (True), or overwrite (False)
# 	consolidated: Export consolidated classes instead
# 	mosaic_mean: Channel means
#	mosaic_std: Channl std
#	bandstoexport: specific bands to export from raster
#	exporttype: gdal.GDT_##### types
#	label_cmap: cmap IN ORDER of export_class_dict

	def export_segmentation_map(self, exporttrainpath, exportlabelpath, txtfilename, image_size=25, N=20000, 
		lastchannelremove = True, labelkey = None, subdir=False, cont = False, consolidated = False, mosaic_mean = 0, mosaic_std = 1, 
		bandstoexport=None, exporttype = gdal.GDT_Float32, label_cmap=None):
		crop_len = int(np.floor(image_size/2))

		if cont:
			f = open(exporttrainpath+txtfilename,'a')
		else:
			f = open(exporttrainpath+txtfilename,'w')

		counter = 0
		classcounter = 0

		if consolidated:
			export_class_dict = self.consolidated_class_dict
			truthimage = self.truthimage_consolidated
			truthcrop = self.truthimage_consolidated[crop_len:self.truthimage_consolidated.shape[0]-crop_len, crop_len:self.truthimage_consolidated.shape[1]-crop_len]
			num_classes = self.consolidated_num_classes
		else:
			export_class_dict = self.class_dict
			truthimage = self.truthimage
			truthcrop = self.truthimage[crop_len:self.truthimage.shape[0]-crop_len, crop_len:self.truthimage.shape[1]-crop_len]
			num_classes = self.num_classes

		for k in export_class_dict:
			[i,j] = np.where(truthcrop == export_class_dict[k])
			if len(i) != 0:
				if len(i) < N:
					idx = [count%len(i) for count in range(N)]
				else:
					idx = np.asarray(random.sample(range(len(i)), N)).astype(int)

				subdirpath = '{}/'.format(k)
				if cont:
					if os.path.exists(exporttrainpath+subdirpath):
						basecount = len([f for f in os.listdir(exporttrainpath+subdirpath)])
					else:
						basecount = 0
				else:
					basecount = 0

				for nn in range(len(idx)):
					# Note: i,j are off of truthcrop, and hence when taken against image needs only +image_size to be centered
					tempimage = self.image[i[idx[nn]]:i[idx[nn]]+image_size, j[idx[nn]]:j[idx[nn]]+image_size, :]
					temptruthimage = truthimage[i[idx[nn]]:i[idx[nn]]+image_size, j[idx[nn]]:j[idx[nn]]+image_size]
					if label_cmap is not None:
						templabel = np.zeros((temptruthimage.shape[0], temptruthimage.shape[1], 3))
						counter2 = 0
						for key in export_class_dict:
							templabel[temptruthimage == export_class_dict[key]] = np.asarray(label_cmap(counter2)[-2::-1])*255 # 8bit-based cmap
							counter2 += 1
						templabel = templabel.astype(np.uint8)
					else:
						templabel = np.asarray(temptruthimage*(255/num_classes)).astype(np.uint8) # Scale evenly classes from 0-255 in grayscale

					if lastchannelremove:
						tempimage = np.delete(tempimage, -1,-1) # Remove last dimension of array

					if self.load_type == "raster":
						trainstr = k + '_' + str(basecount+nn).zfill(8) + '.tif'
						truthstr = k + '_' + str(basecount+nn).zfill(8) + '.tif'
					else:
						trainstr = k + '_' + str(basecount+nn).zfill(8) + '.png'
						truthstr = k + '_' + str(basecount+nn).zfill(8) + '.png'
					# else:
					# 	if self.load_type == "raster":
					# 		trainstr = 'class' + str(k) + '_' + str(nn).zfill(8) + '.tif'
					# 		truthstr = 'class' + str(k) + '_' + str(nn).zfill(8) + '.tif'
					# 	else:
					# 		trainstr = 'class' + str(k) + '_' + str(nn).zfill(8) + '.png'
					# 		truthstr = 'class' + str(k) + '_' + str(nn).zfill(8) + '.png'

					if subdir:
						# else:
						# 	subdirpath = 'class' + str(k) + '/'

						if not os.path.exists(exporttrainpath+subdirpath):
							os.makedirs(exporttrainpath+subdirpath)
						if not os.path.exists(exportlabelpath+subdirpath):
							os.makedirs(exportlabelpath+subdirpath)

						if self.load_type == "raster":
							driver = gdal.GetDriverByName('GTiff')
							if bandstoexport is not None:
								dataset = driver.Create(exporttrainpath+subdirpath+trainstr, image_size, image_size, len(bandstoexport), exporttype)
							else:
								dataset = driver.Create(exporttrainpath+subdirpath+trainstr, image_size, image_size, self.image.shape[2], exporttype)
							x, y = self._calculate_corner(self.geotransform, j[idx[nn]]-crop_len, i[idx[nn]]-crop_len) # calculate the x,y coordinates
							dataset.SetGeoTransform((x, self.geotransform[1], 0, y, 0, self.geotransform[5]))	# set geotransform at x,y coordinates
							dataset.SetProjection(self.projection)

							if bandstoexport is not None:
								counter2 = 0
								for chan in bandstoexport:
									tempchannel = (tempimage[:,:,chan-1] - mosaic_mean[counter2])/mosaic_std[counter2]
									tempchannel[tempchannel > 255] = 255			# Artificial ceiling of 255 for RGB ONLY!
									tempchannel[tempchannel < 0] = 0
									dataset.GetRasterBand(counter2+1).WriteArray(tempchannel)
									dataset.FlushCache()
									counter2 += 1
							else:
								for chan in range(self.image.shape[2]):
									dataset.GetRasterBand(chan+1).WriteArray((tempimage[:,:,chan] - mosaic_mean[chan])/mosaic_std[chan])
									dataset.FlushCache()
							cv2.imwrite(exportlabelpath+subdirpath+truthstr, templabel)
						else:
							cv2.imwrite(exporttrainpath+subdirpath+trainstr, tempimage)
							cv2.imwrite(exportlabelpath+subdirpath+truthstr, templabel)
						f.write('./' + subdirpath+trainstr + ' ' + self.imagefilename + ' ' + str(i[idx[nn]])+' '+str(j[idx[nn]]) + '\n')
					else:
						cv2.imwrite(exporttrainpath+trainstr, tempimage)
						cv2.imwrite(exportlabelpath+truthstr, templabel)
						f.write('./' + trainstr + ' ' + self.imagefilename + ' ' + str(i[idx[nn]])+' '+str(j[idx[nn]]) + '\n')
					print(str(counter*N+nn+1) + '/ ' + str(len(export_class_dict)*N) +' patches exported', end='\r')
				classcounter += 1
			counter += 1
		print("{} of {} total classes found and saved".format(classcounter, len(export_class_dict)))
		f.close()

#### Load entire line(s) of patches from testimage
# Input:
#	image_size: size of image patch
# 	crop_len: sides to crop (if predicting upon middle pixel)
# 	offset: offset from top of image
# 	yoffset: offset from left of image 
# 	cols: number of columns to output
# 	lines: Number of lines of patches to output (None defaults to all possible)
# 	toremove: Remove last channel or not
# Output:
# 	whole_dataset: Patch(es) of test image
	def _load_whole_data(self, image_size, crop_len, offset=0, yoffset = 0, cols = 1, lines=None, toremove=None):
		if image_size%2 == 0:
			if lines is None: 	# this is never used currently
				lines = self.testimage.shape[0] - 2*crop_len +1

			if offset+lines+crop_len > self.testimage.shape[0]+1: # this is never used currently
				print("Too many lines specified, reverting to maximum possible")
				lines = self.testimage.shape[0] - offset - crop_len

			whole_datasets = []
			for i in range(offset, lines+offset):
				#for j in range(crop_len, self.testimage.shape[1] - crop_len):
				for j in range(yoffset, yoffset+cols):
					whole_datasets.append(self.testimage[i-crop_len:i+crop_len, j-crop_len:j+crop_len,:])
		else:
			if lines is None:
				lines = self.testimage.shape[0] - 2*crop_len

			if offset+lines+crop_len > self.testimage.shape[0]:
				print("Too many lines specified, reverting to maximum possible")
				lines = self.testimage.shape[0] - offset - crop_len

			whole_datasets = []
			for i in range(offset, lines+offset):
				for j in range(yoffset, yoffset+cols):
					whole_datasets.append(self.testimage[i-crop_len:i+crop_len+1, j-crop_len:j+crop_len+1,:])

		whole_datasets = np.asarray(whole_datasets)

		if toremove is not None:
			whole_datasets = np.delete(whole_datasets, toremove, -1)
		whole_dataset = self._rescale(whole_datasets)
		return whole_dataset

#### Load entire line(s) of patches from testimage
# Input:
#	model: Keras model to predict on
# 	image_size: size of each image
# 	num_lines: number of lines to predict
# 	spacing: Spacing between each image (row,col)
# 	predict_size: size of prediction area square (starting from center); FCN will use predict_size = image_size
# 	lastchannelremove: Remove last channel or not
# Output:
# 	whole_predict: Predicted class array
#   num_predict: Number of times per prediction in array
# 	prob_predict: Probability of each class per pixel, as calculated by softmax
# 	truth_predict: Original truth image (cropped)
# 	accuracy: Overall accuracy of entire prediction
	def predict_on_whole_image(self, model, image_size, num_classes, num_lines = None, spacing = (1,1), predict_size = 1, lastchannelremove = True):
		crop_len = int(np.floor(image_size/2)) # lengths from sides to not take into account in the calculation of num_lines
		offstart = crop_len-int(np.floor(predict_size/2))

		if image_size%spacing[0] != 0 or image_size%spacing[1] != 0:
			print("Error: Spacing does not divide into image size!")
			raise ValueError

		if image_size%2 == 0:
			if num_lines is None:
				num_lines = int(np.floor((self.testimage.shape[0] - image_size)/spacing[0])+1) # Predict on whole image

			whole_predict = np.zeros((spacing[0]*(num_lines-1)+predict_size, self.testimage.shape[1]-image_size+predict_size))
			num_predict = np.zeros((spacing[0]*(num_lines-1)+predict_size, self.testimage.shape[1]-image_size+predict_size))
			prob_predict = np.zeros((spacing[0]*(num_lines-1)+predict_size, self.testimage.shape[1]-image_size+predict_size, num_classes))

			truth_predict = self.truthimage[offstart:offstart+whole_predict.shape[0], offstart:offstart+whole_predict.shape[1]]

			for offset in range(crop_len, crop_len+spacing[0]*num_lines, spacing[0]):
				for cols in range(crop_len, self.testimage.shape[1]-crop_len+1, spacing[1]):
					if lastchannelremove:
						temp_dataset = self._load_whole_data(image_size, crop_len, offset=offset, yoffset = cols, cols=1, lines=1, toremove=3)
					else:
						temp_dataset = self._load_whole_data(image_size, crop_len, offset=offset, yoffset = cols, cols=1, lines=1)
					temp_prob_predict = model.predict_on_batch(temp_dataset)
					# print(temp_prob_predict.shape)
					temp_predict = self._classifyback(temp_prob_predict)
					
					for predict_mat in temp_predict: 	# this is incorrect if temp_predict has more than 1 prediction (e.g. cols>1, lines>1)
						whole_predict[offset-crop_len:offset-crop_len+predict_size, cols-crop_len:cols-crop_len+predict_size] = \
							whole_predict[offset-crop_len:offset-crop_len+predict_size, cols-crop_len:cols-crop_len+predict_size] + np.reshape(predict_mat, (predict_size,predict_size))
						num_predict[offset-crop_len:offset-crop_len+predict_size, cols-crop_len:cols-crop_len+predict_size] = \
							num_predict[offset-crop_len:offset-crop_len+predict_size, cols-crop_len:cols-crop_len+predict_size] + np.ones((predict_size,predict_size))
						prob_predict[offset-crop_len:offset-crop_len+predict_size, cols-crop_len:cols-crop_len+predict_size,:] = \
							prob_predict[offset-crop_len:offset-crop_len+predict_size, cols-crop_len:cols-crop_len+predict_size,:] + np.reshape(temp_prob_predict, (predict_size,predict_size,num_classes))
					print("Line: " + str(offset-crop_len) + " Col: " + str(cols-crop_len) + '/ ' + str(self.testimage.shape[1]-image_size+1) + ' completed', end='\r')
		else:
			if num_lines is None:
				num_lines = int(np.floor((self.testimage.shape[0] - image_size)/spacing[0])+1) # Predict on whole image

			whole_predict = np.zeros((spacing[0]*(num_lines-1)+predict_size, self.testimage.shape[1]-image_size+predict_size))
			num_predict = np.zeros((spacing[0]*(num_lines-1)+predict_size, self.testimage.shape[1]-image_size+predict_size))

			truth_predict = self.truthimage[offstart:offstart+whole_predict.shape[0], offstart:offstart+whole_predict.shape[1]]

			for offset in range(crop_len, crop_len+spacing[0]*num_lines, spacing[0]):
				for cols in range(crop_len, self.testimage.shape[1]-crop_len, spacing[1]):
					if lastchannelremove:
						temp_dataset = self._load_whole_data(image_size, crop_len, offset=offset, yoffset = cols, cols=1, lines=1, toremove=3)
					else:
						temp_dataset = self._load_whole_data(image_size, crop_len, offset=offset, yoffset = cols, cols=1, lines=1)
					temp_predict = model.predict_on_batch(temp_dataset)
					temp_predict = self._classifyback(temp_predict)
					
					for predict_mat in temp_predict: 	# this is incorrect if temp_predict has more than 1 prediction (e.g. cols>1, lines>1)
						whole_predict[offset-crop_len:offset-crop_len+predict_size, cols-crop_len:cols-crop_len+predict_size] = whole_predict[offset-crop_len:offset-crop_len+predict_size, cols-crop_len:cols-crop_len+predict_size] + predict_mat
						num_predict[offset-crop_len:offset-crop_len+predict_size, cols-crop_len:cols-crop_len+predict_size] = num_predict[offset-crop_len:offset-crop_len+predict_size, cols-crop_len:cols-crop_len+predict_size] + np.ones(predict_mat.shape)
					print("Line: " + str(offset-crop_len) + " Col: " + str(cols-crop_len) + '/ ' + str(self.testimage.shape[1]-2*crop_len) + ' completed', end='\r')

		# Remaining code is for the special case in which spacing does not get to the last col/row

			# Classify last column of row
			# tempcol = whole_predict.shape[1]-image_size
			# temp_dataset = self._load_whole_data(image_size, crop_len, offset=offset, yoffset = tempcol, cols=1, lines=1, toremove = 3)
			# temp_predict = model.predict_on_batch(temp_dataset)
			# temp_predict = self._classifyback(temp_predict)
			# whole_predict[offset:offset+image_size, tempcol:tempcol+image_size] = whole_predict[offset:offset+image_size, tempcol:tempcol+image_size] + predict_mat
			# num_predict[offset:offset+image_size, tempcol:tempcol+image_size] = num_predict[offset:offset+image_size, tempcol:tempcol+image_size] + np.ones(predict_mat.shape)
		
		# Predict on last rows
		# if num_lines*spacing[0]+image_size == int(np.floor((self.testimage.shape[0]-image_size)/spacing[0])):
		# 	offset = self.testimage.shape[0]-image_size
		# 	for cols in range(0, whole_predict.shape[1]-image_size+1, spacing[1]):
		# 		if lastchannelremove:
		# 			temp_dataset = self._load_whole_data(image_size, crop_len, offset=offset, yoffset = cols, cols=1, lines=1, toremove=3)
		# 		else:
		# 			temp_dataset = self._load_whole_data(image_size, crop_len, offset=offset, yoffset = cols, cols=1, lines=1)
		# 		temp_predict = model.predict_on_batch(temp_dataset)
		# 		temp_predict = self._classifyback(temp_predict)
				
		# 		for predict_mat in temp_predict:
		# 			whole_predict[offset:offset+image_size, cols:cols+image_size] = whole_predict[offset:offset+image_size, cols:cols+image_size] + predict_mat
		# 			num_predict[offset:offset+image_size, cols:cols+image_size] = num_predict[offset:offset+image_size, cols:cols+image_size] + np.ones(predict_mat.shape)
		# 		print("Line: " + str(offset) + " Col: " + str(cols) + '/ ' + str(whole_predict.shape[1]-image_size+1) + ' completed', end='\r')

		# 	# Classify last column of row
		# 	tempcol = whole_predict.shape[1]-image_size
		# 	temp_dataset = self._load_whole_data(image_size, crop_len, offset=offset, yoffset = tempcol, cols=1, lines=1, toremove = 3)
		# 	temp_predict = model.predict_on_batch(temp_dataset)
		# 	temp_predict = self._classifyback(temp_predict)
		# 	whole_predict[offset:offset+image_size, tempcol:tempcol+image_size] = whole_predict[offset:offset+image_size, tempcol:tempcol+image_size] + predict_mat
		# 	num_predict[offset:offset+image_size, tempcol:tempcol+image_size] = num_predict[offset:offset+image_size, tempcol:tempcol+image_size] + np.ones(predict_mat.shape)
		
		# whole_predict = np.round(whole_predict.astype(np.float)/num_predict.astype(np.float)).astype(np.uint8)
		# class_dict_min = np.min([self.class_dict[k] for k in self.class_dict])
		whole_predict = np.round(whole_predict.astype(np.float)/num_predict.astype(np.float))
		prob_predict = prob_predict/np.dstack([num_predict.astype(np.float)]*num_classes)
		accuracy = 100*np.asarray(whole_predict == truth_predict).astype(np.float32).sum()/(whole_predict.shape[0]*whole_predict.shape[1]) # this is not correct when truth_predict does not start from 0

		return whole_predict, num_predict, prob_predict, truth_predict, accuracy

# Input:
	#   cm: confusion matrix from sklearn: confusion_matrix(y_true, y_pred)
	#   classes: a list of class labels
	#   normalize: whether the cm is shown as number of percentage (normalized)
	#   file2sav: unique filename identifier in case of multiple cm
	# Output:
	#   .png plot of cm
def plot_confusion_matrix(cm, classes, normalize=False,
					title='Confusion matrix',
					cmap=plt.cm.Blues, plot_colorbar = False, file2sav = "cm"):
	#"""
	#This function prints and plots the confusion matrix.
	#Normalization can be applied by setting `normalize=True`.
	#cm can be a unique file identifier if multiple options exist
	#"""
	if normalize:
		cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
		print("Normalized confusion matrix")
		tit = "normalized"
	else:
		print('Confusion matrix, without normalization')
		tit = "non_normalized"
	# print(cm)

	cm_plot = './plots/Confusion_Matrix_' + tit + file2sav + ".png"
	plt.figure()
	plt.imshow(cm, interpolation='nearest', cmap=cmap)
	plt.title(title + " " + tit)
	if plot_colorbar:
		plt.colorbar(fraction=0.046, pad=0.04)
	tick_marks = np.arange(len(classes))
	plt.xticks(tick_marks, classes, rotation=90,fontsize=16)
	plt.yticks(tick_marks, classes, fontsize=16)

	fmt = '.2f' if normalize else 'd'
	if normalize:
		thresh = 0.5
	else:
		thresh = cm.max() / 2.
	for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
		plt.text(j, i, format(cm[i, j], fmt),
			horizontalalignment="center", fontsize=16, 
			color="white" if cm[i, j] > thresh else "black")
		#plt.text(j, i, r'{0:.2f}'.format(cm[i,j]), 
	#		horizontalalignment="center", fontsize=16, 
	#		color="white" if cm[i, j] > thresh else "black")

	#plt.tight_layout()
	plt.ylabel('True label', fontsize=16)
	plt.xlabel('Predicted label', fontsize=16)
	fig = plt.gcf()
	fig.set_size_inches(20, 20, forward=True)
	plt.show()
	# plt.savefig(cm_plot)
	# plt.close()


def confusion_matrix_stats(cm, classes, file2sav = "cm_stats"):
	#"""
	#This function calculates stats related to the confusion matrix cm
	#cm - confusion matrix as numpy array, can be generated or loaded (cm = np.load('./output/cm_whole_image_hyperas1.npy'))
	#classes - a list of class labels
	#file2save - filename (without csv extension)
	#"""
	TP = np.trace(cm)
	sum_pred = cm.sum(axis=0) # summing over predicted values (each columns going over all rows)
	sum_true = cm.sum(axis=1) # summing over true values (each row going over all columns)

	total_pred = sum_pred.sum()
	total_true = sum_true.sum()

	overall_accuracy = (float(TP) / float(total_true))*100.
	print("overall_accuracy: " + str(np.round(overall_accuracy, decimals=2)) + "%")

	diag = cm.diagonal()
	prod_acc = np.true_divide((diag), (sum_true))
	user_acc = np.true_divide((diag), (sum_pred))
	
	d = {'class_label': label_list, 'prod_acc': prod_acc, 'user_acc': user_acc, 'overall_acc': overall_accuracy/100.}
	df = pd.DataFrame(data=d)
	
	# df.to_csv('./output/' +file2save + '.csv')

# fill in last remaining color on a truthmap
# truthmap_fn: path to truthmap
# cmap: colormap of colors that already exist in the truthmap
# lastcolor: last color to fill in, in BGR
def fill_in_truthmap_lastcolor(truthmap_fn, cmap, lastcolor):
	truthmap = cv2.imread(truthmap_fn) 	# Read in as BGR
	cmap_8bit = np.asarray([np.asarray(cmap(i)[-2::-1])*255 for i in range(len(cmap.colors))], dtype = np.uint8)	 #in BGR

	replacemap = np.ones((truthmap.shape[0], truthmap.shape[1]), dtype=np.bool)
	for color in cmap_8bit:
		idx = np.where(np.all(truthmap==color, axis=-1))
		replacemap[idx] = False
	truthmap[replacemap] = lastcolor
	return truthmap

# fill in empty pixels based upon surrounding pixel colors
# truthmap_fn: path to truthmap (classified from people)
# surroundingarea: square grid size arround classification pixel (pick an odd number!)
def fill_in_truthmap(truthmap_fn, surroundingarea):
	white = np.asarray([255,255,255])
	if surroundingarea % 2 == 0:
		raise ValueError('Please choose an odd number for fill_in_truthmap surroundingarea')
	truthmap = cv2.imread(truthmap_fn)
	y,x = np.where(np.all(truthmap == white, axis=-1))
	for j,i in zip(y,x):
		crop_len = int((surroundingarea-1)/2)
		found_replace = False
		while found_replace is False:
			tempy_min = max(j-crop_len,0)
			tempy_max = min(j+crop_len+1,truthmap.shape[0])
			tempx_min = max(i-crop_len,0)
			tempx_max = min(i+crop_len+1, truthmap.shape[1])
			truthmap_patch = truthmap[tempy_min:tempy_max,tempx_min:tempx_max,:]
			unq, unq_count = np.unique(truthmap_patch.reshape(-1, truthmap_patch.shape[2]), return_counts=True, axis=0)
			idx = np.where(np.all(unq == white, axis=-1))

			if len(idx[0]) > 0: 		# Get rid of white counts
				unq = np.delete(unq, idx, axis=0)
				unq_count = np.delete(unq_count, idx, axis=0)

			if len(unq) > 0:			# Make sure there is still at least 1 unique left
				found_replace = True
			else:						# If no uniques left, increment area by 1
				crop_len += 1
		maxidx = np.argmax(unq_count)
		truthmap[j,i] = unq[maxidx]
	return truthmap

# Loads patch from mosaiced .tif file
# imagepath: file where raster file is located
# specific_fn: specific patch filename of the NxN patch being loaded
# trainfile: file where all info is stored for all NxN patches
# image_size: Size of image (just 1 number since it is a square)
# offset: (y,x) offset of patch to load in tuple
def load_specific_patch(imagepath, specific_fn, trainfile, image_size, offset=0):
	f = open(trainfile,"r")
	col = []
	row = []
	rastername = []
	patch_name = []
	for line in f:
		infosplit = line.split(" ")
		col.append(int(infosplit[-1]))
		row.append(int(infosplit[-2]))
		rastername.append(infosplit[-3])
		patch_path = ' '.join(infosplit[0:-3])
		head, tail = os.path.split(patch_path)
		patch_name.append(tail)

	idx = patch_name.index(specific_fn)
	raster = CoralData(imagepath+'/'+rastername[idx], load_type="raster")
	patch = raster.image[row[idx]+offset:row[idx]+offset+image_size, col[idx]+offset:col[idx]+offset+image_size, :]
	projection = raster.projection
	# geotransform is organized as [top left x, w-e pixel resolution, 0, top left y, 0, n-s pixel resolution (negative)]
	# print(raster.geotransform)
	geotransform = [g for g in raster.geotransform]
	# geotransform = [(g[0]+offset*g[1], g[1], g[2], g[3]+offset*g[5], g[4], g[5]) for g in raster.geotransform]
	# geotransform = raster.geotransform
	geotransform[0] = geotransform[0] + offset*geotransform[1]
	geotransform[3] = geotransform[3] + offset*geotransform[5]

	return patch, projection, geotransform

# Turns RGB cmap into dictionary-defined truthmap
# Note: Dictionary might go from 1 to num_classes, and hence the grayscale truthmap will go from num_classes:255/num_classes:255
# RGBpath: path to all RGB truthmaps
# Graypath: Final path to put all Grayscale truthmaps
# cmap: RGB cmap
# class_dict: Dictionary associated with each color to class (make sure it's in same order as cmap!)
def transform_RGB2Gray(RGBpath, Graypath, cmap, class_dict):
	files = [f for f in os.listdir(RGBpath) if os.path.isfile(os.path.join(RGBpath,f))]
	classvalues = [class_dict[k] for k in class_dict]
	cmap_len = len(cmap.colors)
	for f in files:
		BGRpatch = cv2.imread(os.path.join(RGBpath,f))
		Graypatch = np.zeros((BGRpatch.shape[0],BGRpatch.shape[1]), dtype=np.uint8)
		for i in range(cmap_len):
			y,x = np.where(np.all(BGRpatch == np.asarray(np.asarray(cmap(i)[-2::-1])*255, dtype=np.uint8), axis=-1))
			Graypatch[y,x] = np.uint8(classvalues[i]*255/cmap_len)
		cv2.imwrite(os.path.join(Graypath,f), Graypatch)