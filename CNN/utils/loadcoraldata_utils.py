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
		self.load_type = load_type
		if load_type == "PIL":
			self.image = img_to_array(pil_image.open(Imagepath))
			if Truthpath is not None:
				self.truthimage = cv2.imread(Truthpath, cv2.IMREAD_UNCHANGED)
			# if Testpath is not None:
			# 	self.testimage = img_to_array(pil_image.open(Testpath))
		elif load_type == "cv2":
			self.image = cv2.imread(Imagepath,cv2.IMREAD_UNCHANGED)
			if Truthpath is not None:
				self.truthimage = cv2.imread(Truthpath,cv2.IMREAD_UNCHANGED)
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
			if Truthpath.endswith('.tif'):
				self.truthimage = cv2.imread(Truthpath,cv2.IMREAD_UNCHANGED)
				class_indices = np.unique(self.truthimage)
				num_classes = len(class_indices)
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

				gdal_truthimg = None
		if Bathypath is not None:
			self.bathyimage = cv2.imread(Bathypath, cv2.IMREAD_UNCHANGED)
			self.bathyimage[self.bathyimage == np.min(self.bathyimage)] = -1

	    # Set labels from 0 to item_counter based upon input truth_key
		if truth_key is not None:
			item_counter = 0
			for item in truth_key:
				self.truthimage[self.truthimage == item] = item_counter
				item_counter+=1
			self.num_classes = len(np.unique(self.truthimage))
		else:
			try:
				self.num_classes = len(self.class_labels)
			except:
				pass

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

#### Generate a randomized, normalized set of images/labels from original image + reference labels
# Note that the label data is only the central pixel label of the defined training image
# Input:
# 	image_size: Size of image
# 	n: Number of images per class
# 	idxremove: Which index to remove (NOTE: This is NOT the channel index, but is from the index of a list of datasets)
# 	datasets: list of images, starts as empty []
# 	labels: list of labels: starts as empty []
# 	figureson: True/False, displays 1 image from each class
# Output:
# 	datasets: set of randomized, normalized images, N_images x nrow x ncol x n_channels
# 	labels: set of randomized labels (concurrent with datasets), N_images x num_labels (num_labels = 1 for now)
	def _generate_randomized_set(self, image_size, n, idxremove, datasets, labels, figureson):
		crop_len = int(np.floor(image_size/2))
		try:
			# Crop the image so that border areas are not considered
			truthcrop = self.truthimage[crop_len:self.truthimage.shape[0]-crop_len, crop_len:self.truthimage.shape[1]-crop_len]
		except TypeError:
			print("Truth/Reference image improperly defined")
			return

		for k in range(self.num_classes):
			[i,j] = np.where(truthcrop == k)
			idx = np.asarray(random.sample(range(len(i)), n)).astype(int)
			datasets[k*n:(k+1)*n,:,:,:] = [self.image[i[idx[nn]]:i[idx[nn]]+image_size, j[idx[nn]]:j[idx[nn]]+image_size, :] for nn in range(len(idx))]
			labels[k*n:(k+1)*n,0] = [truthcrop[i[idx[nn]], j[idx[nn]]] for nn in range(len(idx))]
			# datasets.append([self.image[i[idx[nn]]:i[idx[nn]]+image_size, j[idx[nn]]:j[idx[nn]]+image_size, :] for nn in range(len(idx))])
			# labels.append([truthcrop[i[idx[nn]], j[idx[nn]]] for nn in range(len(idx))])

		# datasets = np.asarray(datasets) # train_datasets is in the format of num_labels x N_train x nrows x ncols x n_channels
		# labels = np.asarray(labels) # train_labels is in the format of num_labels x N_train
		# datasets = datasets.reshape(self.num_classes*n, image_size, image_size, self.image.shape[-1]) # flatten first 2 dimensions of train_datasets
		# labels = labels.reshape(self.num_classes*n,1) # flatten into vector

		if idxremove is not None:
			datasets = np.delete(datasets,idxremove,-1) # Remove specific last dimension of array

		if figureson:
			plt.figure()
			for i in range(self.num_classes):
				plt.subplot(1, self.num_classes, i+1)
				plt.imshow(datasets[i*n,:,:,0:3].astype(np.uint8))
				# plt.imshow(cv2.cvtColor(datasets[i*n,:,:,0:3], cv2.COLOR_BGR2RGB))
				if i > 0:
					plt.axis("off")
			plt.show()

		datasets, labels = self._randomize(datasets, labels)
		datasets = self._rescale(datasets)
		return datasets, labels

# Input:
#	exporttrainpath: Directory for exported patch images 
# 	exportlabelpath: Directory for exported segmented images
# 	txtfilename: Name of text file to record image names (remember to include '.txt')
# 	image_size: Size of image
# 	N: Number of images per class (NOTE: because these are segmented maps, the class is only affiliated with the center pixel)
# 	lastchannelremove: Remove last channel or not
# 	labelkey: Naming convention of class labels (NOTE: must be same # as the # of classes)
# 	subdir: Create subdirectories for each class
	def export_segmentation_map(self, exporttrainpath, exportlabelpath, txtfilename, image_size=25, N=20000, lastchannelremove = True, labelkey = None, subdir=False):
		crop_len = int(np.floor(image_size/2))
		try:
			# Crop the image so that border areas are not considered
			truthcrop = self.truthimage[crop_len:self.truthimage.shape[0]-crop_len, crop_len:self.truthimage.shape[1]-crop_len]
		except TypeError:
			print("Truth/Reference image improperly defined")
			return

		f = open(exporttrainpath+txtfilename,'w')

		for k in range(self.num_classes):
			[i,j] = np.where(truthcrop == k)
			if len(i) != 0:
				if len(i) < N:
					idx = [count%len(i) for count in range(N)]
				else:
					idx = np.asarray(random.sample(range(len(i)), N)).astype(int)

				for nn in range(len(idx)):
					# Note: i,j are off of truthcrop, and hence when taken against image needs only +image_size to be centered
					tempimage = self.image[i[idx[nn]]:i[idx[nn]]+image_size, j[idx[nn]]:j[idx[nn]]+image_size, :]
					templabel = np.asarray(self.truthimage[i[idx[nn]]:i[idx[nn]]+image_size, j[idx[nn]]:j[idx[nn]]+image_size]*(255/self.num_classes)).astype(np.uint8)

					if lastchannelremove:
						tempimage = np.delete(tempimage, -1,-1) # Remove last dimension of array

					if labelkey is not None:
						if self.load_type == "raster":
							trainstr = labelkey[k] + '_' + str(nn).zfill(8) + '.tif'
							truthstr = labelkey[k] + '_' + str(nn).zfill(8) + '.tif'
						else:
							trainstr = labelkey[k] + '_' + str(nn).zfill(8) + '.png'
							truthstr = labelkey[k] + '_' + str(nn).zfill(8) + '.png'
					else:
						if self.load_type == "raster":
							trainstr = 'class' + str(k) + '_' + str(nn).zfill(8) + '.tif'
							truthstr = 'class' + str(k) + '_' + str(nn).zfill(8) + '.tif'
						else:
							trainstr = 'class' + str(k) + '_' + str(nn).zfill(8) + '.png'
							truthstr = 'class' + str(k) + '_' + str(nn).zfill(8) + '.png'

					if subdir:
						if labelkey is not None:
							subdirpath = '{}/'.format(labelkey[k])
						else:
							subdirpath = 'class' + str(k) + '/'

						if not os.path.exists(exporttrainpath+subdirpath):
							os.makedirs(exporttrainpath+subdirpath)
						if not os.path.exists(exportlabelpath+subdirpath):
							os.makedirs(exportlabelpath+subdirpath)

						if self.load_type == "raster":
							driver = gdal.GetDriverByName('GTiff')
							dataset = driver.Create(exporttrainpath+subdirpath+trainstr, image_size, image_size, self.image.shape[2], gdal.GDT_Int32)
							x, y = self._calculate_corner(self.geotransform, j[idx[nn]]-crop_len, i[idx[nn]]-crop_len)
							# print(x, self.geotransform[1], y, self.geotransform[5])
							dataset.SetGeoTransform((x, self.geotransform[1], 0, y, 0, self.geotransform[5]))
							dataset.SetProjection(self.projection)

							for chan in range(self.image.shape[2]):
								dataset.GetRasterBand(chan+1).WriteArray(tempimage[:,:,chan])
								dataset.FlushCache()
							cv2.imwrite(exportlabelpath+subdirpath+truthstr, templabel)
						else:
							cv2.imwrite(exporttrainpath+subdirpath+trainstr, tempimage)
							cv2.imwrite(exportlabelpath+subdirpath+truthstr, templabel)
						f.write('./' + subdirpath+trainstr+'\n')
					else:
						cv2.imwrite(exporttrainpath+trainstr, tempimage)
						cv2.imwrite(exportlabelpath+truthstr, templabel)
						f.write('./' + trainstr+'\n')
					print(str(k*N+nn+1) + '/ ' + str(self.num_classes*N) +' patches exported', end='\r')
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
		whole_predict = np.round(whole_predict.astype(np.float)/num_predict.astype(np.float))
		prob_predict = prob_predict/np.dstack([num_predict.astype(np.float)]*num_classes)
		accuracy = 100*np.asarray((whole_predict == truth_predict)).astype(np.float32).sum()/(whole_predict.shape[0]*whole_predict.shape[1])

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


