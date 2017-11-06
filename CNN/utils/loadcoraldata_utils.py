import numpy as np
import cv2
import random
from matplotlib import pyplot as plt

# Class of coral data, consisting of an image and possibly a corresponding truth map
class CoralData:
	image = None
	num_classes = 0
	truthimage = None
	train_datasets = None
	train_labels = None
	valid_datasets = None
	valid_labels = None
	test_datasets = None
	test_labels = None
	depth = 255

	def __init__(self, Imagepath, Truthpath=None, truth_key=None):
		# Load images
	    self.image = cv2.imread(Imagepath,cv2.IMREAD_UNCHANGED)
	    if Truthpath is not None:
	    	self.truthimage = cv2.imread(Truthpath,cv2.IMREAD_UNCHANGED)
	    # Set labels from 0 to item_counter based upon input truth_key
	    if truth_key is not None:
	    	item_counter = 0
	    	for item in truth_key:
	    		self.truthimage[self.truthimage == item] = item_counter
	    		item_counter+=1
	    	self.num_classes = len(np.unique(self.truthimage))

#### Load Image
# Input:
# 	Imagepath: Path to Image
	def load_image(self, Imagepath):
		self.image = cv2.imread(Imagepath,cv2.IMREAD_UNCHANGED)

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

#### Normalize Image
# Input:
# 	dataset: set of vectorized images, N_images x nrow x ncol x n_channels
# Output:
# 	Set of vectorized normalized images, N_images x nrow x ncol x n_channels
	def _rescale(self, dataset):
		return (dataset.astype(np.float32) - self.depth/2)/(self.depth/2)

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


	def export_segmentation_map(self, exporttrainpath, exportlabelpath, image_size=25, N=20000, lastchannelremove = True, labelkey = None):
		crop_len = int(np.floor(image_size/2))
		try:
			# Crop the image so that border areas are not considered
			truthcrop = self.truthimage[crop_len:self.truthimage.shape[0]-crop_len, crop_len:self.truthimage.shape[1]-crop_len]
		except TypeError:
			print("Truth/Reference image improperly defined")
			return

		for k in range(self.num_classes):
			[i,j] = np.where(truthcrop == k)
			idx = np.asarray(random.sample(range(len(i)), N)).astype(int)
			for nn in range(len(idx)):
				tempimage = self.image[i[idx[nn]]:i[idx[nn]]+image_size, j[idx[nn]]:j[idx[nn]]+image_size, :]
				templabel = np.asarray(self.truthimage[i[idx[nn]]:i[idx[nn]]+image_size, j[idx[nn]]:j[idx[nn]]+image_size]*(self.depth/self.num_classes)).astype(np.uint8)

				if lastchannelremove:
					tempimage = np.delete(tempimage, -1,-1) # Remove last dimension of array

				if labelkey is not None:
					trainstr = labelkey[k] + '_' + str(nn).zfill(8) + '.tiff'
					truthstr = labelkey[k] + '_' + str(nn).zfill(8) + '.tiff'
				else:
					trainstr = 'class' + str(k) + '_' + str(nn).zfill(8) + '.tiff'
					truthstr = 'class' + str(k) + '_' + str(nn).zfill(8) + '.tiff'

				cv2.imwrite(exporttrainpath+trainstr, tempimage)
				cv2.imwrite(exportlabelpath+truthstr, templabel)