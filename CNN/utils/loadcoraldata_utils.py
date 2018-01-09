import numpy as np
import cv2
import random
import os
import glob
from osgeo import gdal, ogr, osr
from matplotlib import pyplot as plt
from PIL import Image as pil_image
from keras.preprocessing.image import img_to_array

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

	def __init__(self, Imagepath, Truthpath=None, Testpath = None, truth_key=None, load_type="cv2", tfwpath=None):
		# Load images
		self.load_type = load_type
		if load_type == "PIL":
			self.image = img_to_array(pil_image.open(Imagepath))
			if Truthpath is not None:
				self.truthimage = cv2.imread(Truthpath, cv2.IMREAD_UNCHANGED)
			if Testpath is not None:
				self.testimage = img_to_array(pil_image.open(Testpath))
		elif load_type == "cv2":
			self.image = cv2.imread(Imagepath,cv2.IMREAD_UNCHANGED)
			if Truthpath is not None:
				self.truthimage = cv2.imread(Truthpath,cv2.IMREAD_UNCHANGED)
			if Testpath is not None:
				self.testimage = cv2.imread(Testpath, cv2.IMREAD_UNCHANGED)
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
				img_xmin = self.geotransform[0]
				img_ymax = self.geotransform[3]

			self.image = np.zeros((ysize,xsize,img.RasterCount))

			for band in range(img.RasterCount):
				band += 1
				imgband = img.GetRasterBand(band)
				self.image[:,:,band-1] = imgband.ReadAsArray()

			if Truthpath is not None:
				counter = 1
				self.class_labels = ['NoData']
				raster_fn = 'temp.tif'
				NoData_value = -1

				for file in glob.glob(Truthpath + '*.shp'):
					vector_fn = file
					underscore_pos = vector_fn.find('_')
					period_pos = -4
					self.class_labels.append(file[underscore_pos+1:period_pos])
					print("Class loaded: " + file[underscore_pos+1:period_pos] + "\n")

					source_ds = ogr.Open(vector_fn)
					source_layer = source_ds.GetLayer()
					truth_xmin, truth_xmax, truth_ymin, truth_ymax = source_layer.GetExtent()
					pixel_size = self.geotransform[1]

					topleft = (np.max([img_xmin, truth_xmin]), np.min([img_ymax, truth_ymax]))
					bottomright = (np.min([img_xmax, truth_xmax]), np.max([img_ymin, truth_ymin]))

					x_res = int((xmax - xmin) / pixel_size)
					y_res = int((ymax - ymin) / pixel_size)

					target_ds = gdal.GetDriverByName('GTiff').Create(raster_fn, x_res, y_res, 1, gdal.GDT_Int32)
					target_ds.SetGeoTransform(self.geotransform)
					band = target_ds.GetRasterBand(1)
					band.SetNoDataValue(NoData_value)
					gdal.RasterizeLayer(target_ds, [1], source_layer, None, None, [1], ['ALL_TOUCHED=TRUE'])

					tempband = target_ds.GetRasterBand(1)
					temparray = tempband.ReadAsArray()
					temparray[temparray == -1] = 0
					temparray[temparray == 1] = counter

					if self.truthimage is None:
						self.truthimage = np.copy(temparray)
					else:
						temparray2 = temparray + self.truthimage
						temparray[temparray2 > counter] = 0
						self.truthimage = self.truthimage + temparray

					target_ds = None
					counter +=1
				self.truthimage = self.truthimage.astype(np.uint8)

		else:
			print("Load type error: specify either PIL, cv2, or raster")
			return None

	    # Set labels from 0 to item_counter based upon input truth_key
		if truth_key is not None:
			item_counter = 0
			for item in truth_key:
				self.truthimage[self.truthimage == item] = item_counter
				item_counter+=1
			self.num_classes = len(np.unique(self.truthimage))
		else:
			self.num_classes = len(np.unique(self.truthimage))

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

	def generate_trainingset(self, image_size=25, N_train=20000, idxremove = None, figureson = False):
		self.train_datasets = np.zeros(shape=(N_train*self.num_classes, image_size, image_size, self.image.shape[-1]))
		self.train_labels = np.zeros(shape=(N_train*self.num_classes, 1))
		self.train_datasets, self.train_labels = self._generate_randomized_set(image_size, N_train, idxremove, self.train_datasets, self.train_labels, figureson)

	def generate_validset(self, image_size=25, N_valid=2500, idxremove  = None, figureson = False):
		self.valid_datasets = np.zeros(shape=(N_valid*self.num_classes, image_size, image_size, self.image.shape[-1]))
		self.valid_labels = np.zeros(shape=(N_valid*self.num_classes, 1))
		self.valid_datasets, self.valid_labels = self._generate_randomized_set(image_size, N_valid, idxremove , self.valid_datasets, self.valid_labels, figureson)

	def generate_testset(self, image_size=25, N_test=2500, idxremove  = None, figureson = False):
		self.test_datasets = np.zeros(shape=(N_test*self.num_classes, image_size, image_size, self.image.shape[-1]))
		self.test_labels = np.zeros(shape=(N_test*self.num_classes, 1))
		self.test_datasets, self.test_labels = self._generate_randomized_set(image_size, N_test, idxremove , self.test_datasets, self.test_labels, figureson)

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
			idx = np.asarray(random.sample(range(len(i)), N)).astype(int)
			for nn in range(len(idx)):
				# Note: i,j are off of truthcrop, and hence when taken against image needs only +image_size to be centered
				tempimage = self.image[i[idx[nn]]:i[idx[nn]]+image_size, j[idx[nn]]:j[idx[nn]]+image_size, :]
				templabel = np.asarray(self.truthimage[i[idx[nn]]:i[idx[nn]]+image_size, j[idx[nn]]:j[idx[nn]]+image_size]*(self.depth/self.num_classes)).astype(np.uint8)

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
# 	lines: Number of lines of patches to output
# 	toremove: Remove last channel or not
# Output:
# 	whole_dataset: Entire line(s) of patches from testimage
	def _load_whole_data(self, image_size, crop_len, offset=0, yoffset = 0, cols = 1, lines=None, toremove=False):
		if lines is None:
			lines = self.testimage.shape[0] - 2*crop_len

		if offset+lines+2*crop_len > self.testimage.shape[0]:
			print("Too many lines specified, reverting to maximum possible")
			lines = self.testimage.shape[0] - offset - 2*crop_len

		whole_datasets = []
		for i in range(offset+crop_len, lines+offset+crop_len):
			#for j in range(crop_len, self.testimage.shape[1] - crop_len):
			for j in range(yoffset+crop_len, yoffset+crop_len+cols):
				whole_datasets.append(self.testimage[i-crop_len:i+crop_len, j-crop_len:j+crop_len,:])

		whole_datasets = np.asarray(whole_datasets)

		if toremove is not None:
			whole_datasets = np.delete(whole_datasets, toremove, -1)
		whole_dataset = self._rescale(whole_datasets)
		return whole_dataset

	def predict_on_whole_image(self, model, image_size, num_lines = None, spacing = (1,1), crop = False, lastchannelremove = True):
		offstart = 0
		crop_len = int(np.floor(image_size/2)) 	# crop_len is NOT currently used to crop the image!!! 

		if num_lines is None:
			num_lines = int(np.floor((self.testimage.shape[0] - 2*crop_len)/spacing[0]))

		if num_lines*spacing[0]+image_size == int(np.floor((self.testimage.shape[0]-image_size)/spacing[0])): # If predict on whole image
			whole_predict = np.zeros(self.testimage.shape)
			num_predict = np.zeros(self.testimage.shape)
		else:
			whole_predict = np.zeros((spacing[0]*(num_lines-1)+image_size,self.testimage.shape[1]))
			num_predict = np.zeros((spacing[0]*(num_lines-1)+image_size,self.testimage.shape[1]))

		truth_predict = self.truthimage[offstart:offstart+whole_predict.shape[0], 0:whole_predict.shape[1]]

		for offset in range(offstart, offstart+spacing[0]*num_lines, spacing[0]):
			for cols in range(0, whole_predict.shape[1]-image_size+1, spacing[1]):
				if lastchannelremove:
					temp_dataset = self._load_whole_data(image_size, crop_len, offset=offset, yoffset = cols, cols=1, lines=1, toremove=3)
				else:
					temp_dataset = self._load_whole_data(image_size, crop_len, offset=offset, yoffset = cols, cols=1, lines=1)
				temp_predict = model.predict_on_batch(temp_dataset)
				temp_predict = self._classifyback(temp_predict)
				
				for predict_mat in temp_predict:
					whole_predict[offset:offset+image_size, cols:cols+image_size] = whole_predict[offset:offset+image_size, cols:cols+image_size] + predict_mat
					num_predict[offset:offset+image_size, cols:cols+image_size] = num_predict[offset:offset+image_size, cols:cols+image_size] + np.ones(predict_mat.shape)
				print("Line: " + str(offset) + " Col: " + str(cols) + '/ ' + str(whole_predict.shape[1]-image_size+1) + ' completed', end='\r')

			# Classify last column of row
			tempcol = whole_predict.shape[1]-image_size
			temp_dataset = self._load_whole_data(image_size, crop_len, offset=offset, yoffset = tempcol, cols=1, lines=1, toremove = 3)
			temp_predict = model.predict_on_batch(temp_dataset)
			temp_predict = self._classifyback(temp_predict)
			whole_predict[offset:offset+image_size, tempcol:tempcol+image_size] = whole_predict[offset:offset+image_size, tempcol:tempcol+image_size] + predict_mat
			num_predict[offset:offset+image_size, tempcol:tempcol+image_size] = num_predict[offset:offset+image_size, tempcol:tempcol+image_size] + np.ones(predict_mat.shape)
		
		# Predict on last rows
		if num_lines*spacing[0]+image_size == int(np.floor((self.testimage.shape[0]-image_size)/spacing[0])):
			offset = self.testimage.shape[0]-image_size
			for cols in range(0, whole_predict.shape[1]-image_size+1, spacing[1]):
				if lastchannelremove:
					temp_dataset = self._load_whole_data(image_size, crop_len, offset=offset, yoffset = cols, cols=1, lines=1, toremove=3)
				else:
					temp_dataset = self._load_whole_data(image_size, crop_len, offset=offset, yoffset = cols, cols=1, lines=1)
				temp_predict = model.predict_on_batch(temp_dataset)
				temp_predict = self._classifyback(temp_predict)
				
				for predict_mat in temp_predict:
					whole_predict[offset:offset+image_size, cols:cols+image_size] = whole_predict[offset:offset+image_size, cols:cols+image_size] + predict_mat
					num_predict[offset:offset+image_size, cols:cols+image_size] = num_predict[offset:offset+image_size, cols:cols+image_size] + np.ones(predict_mat.shape)
				print("Line: " + str(offset) + " Col: " + str(cols) + '/ ' + str(whole_predict.shape[1]-image_size+1) + ' completed', end='\r')

			# Classify last column of row
			tempcol = whole_predict.shape[1]-image_size
			temp_dataset = self._load_whole_data(image_size, crop_len, offset=offset, yoffset = tempcol, cols=1, lines=1, toremove = 3)
			temp_predict = model.predict_on_batch(temp_dataset)
			temp_predict = self._classifyback(temp_predict)
			whole_predict[offset:offset+image_size, tempcol:tempcol+image_size] = whole_predict[offset:offset+image_size, tempcol:tempcol+image_size] + predict_mat
			num_predict[offset:offset+image_size, tempcol:tempcol+image_size] = num_predict[offset:offset+image_size, tempcol:tempcol+image_size] + np.ones(predict_mat.shape)
		
		whole_predict = np.round(whole_predict.astype(np.float)/num_predict.astype(np.float)).astype(np.uint8)
		accuracy = 100*np.asarray((whole_predict == truth_predict)).astype(np.float32).sum()/(whole_predict.shape[0]*whole_predict.shape[1])

		return whole_predict, num_predict, truth_predict, accuracy


