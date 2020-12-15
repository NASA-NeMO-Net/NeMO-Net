from typing import Tuple, Callable, List, Union, Dict

import numpy as np
import keras
from keras.utils.np_utils import to_categorical
import pydensecrf.densecrf as dcrf
from pydensecrf.utils import unary_from_labels, create_pairwise_bilateral, create_pairwise_gaussian, unary_from_softmax
from sklearn import neighbors
from sklearn.cluster import KMeans, DBSCAN

from NeMO_Utils import apply_channel_corrections, normalize

class CoralPredict:

	def __init__(self, 
		image_array: np.ndarray,
		classes: Dict[str, int],
		label_array: np.ndarray = None,
		patch_size: int = 256,
		spacing: Tuple[int, int] = (256, 256),
		predict_size: int = 256,
		image_mean: Union[float, List] = 0.0,
		image_std: Union[float, List] = 1.0):
		''' image_array: Image array (RGB + NIR, ideally)
			classes: Class dictionary
			label_array: Label "truth" data if available
			patch size: Size of patch that the model outputs (usually 256)
			spacing: Spacing between each patch to predict [row, col]
			predict_size: size of prediction area square (centered). This will disregard predictions of areas that are close to the boundaries.
				 Must be smaller than patch_size.
			image_mean: mean value to normalize by before prediction
			image_std: std value to normalize by before prediction
		'''
		self.image_shape = image_array.shape
		self.image_array = image_array
		self.label_array = label_array
		self.classes = classes
		self.num_classes = len(classes)
		self.patch_size = patch_size
		self.spacing = spacing
		self.predict_size = predict_size

		self.crop_len = int(np.floor(self.patch_size/2))
		self.offstart = self.crop_len - int(np.floor(self.predict_size/2))
		# example: for an image_array of size (256,256), if patch_size = 256, predict_size = 128
		# that means that only image_array[64:192, 64:192] will be predicted on (offstart = 64, crop_len = 128)

		self.image_mean = apply_channel_corrections(image_mean, image_array.shape[2], 0.0, "image_mean")
		self.image_std = apply_channel_corrections(image_std, image_array.shape[2], 1.0, "image_std")

	def predict_on_whole_image(self,
		model: keras.engine.training.Model,
		num_lines: int = None) -> Tuple[np.ndarray, np.ndarray]:
		''' Predicts upon image_array using model, patch-based with overlap
			model: Keras model to predict with
			num_lines: number of rows of patches to predict. If None, then this will be automatically calculated.
			
			Output:
			final_predict: Predicted class array based upon most common predictions
			prob_predict: Probably of each class per pixel, as calculated by softmax
		'''
		image_array = np.copy(self.image_array)
		ysize, xsize = self.image_shape[0], self.image_shape[1]
		offstart = self.offstart
		predict_size = self.predict_size
		spacing = self.spacing
		patch_size = self.patch_size
		crop_len = self.crop_len

		label_array = np.copy(self.label_array)
		label_array = label_array[offstart: ysize - offstart, offstart: xsize - offstart]
           
        # make sure spacing divides into image sizes, so that all pixels will be considered during training
		if (ysize % spacing[0] != 0) or (xsize % spacing[1]) != 0:
			print("Error: Spacing does not divide into image size!")
			raise ValueError
            
		if spacing[0] > predict_size or spacing[1] > predict_size:
			print("Spacing must be smaller than or equal to predict_size!")
			raise ValueError

		if num_lines is None:
			num_lines = int((ysize - patch_size)//spacing[0] + 1) # Calculate number of lines of patches to predict for whole image
		num_cols = int((xsize - patch_size)//spacing[0] + 1)

		# prepare arrays with appropriate sizes
		# Note that we can manually control how many rows we predict, but we will always try to predict max amount of columns
		whole_predict = np.zeros((spacing[0]*(num_lines-1) + predict_size, spacing[1]*(num_cols-1) + predict_size, self.num_classes))
		num_predict = np.zeros((spacing[0]*(num_lines-1) + predict_size, spacing[1]*(num_cols-1) + predict_size))
		prob_predict = np.zeros((spacing[0]*(num_lines-1) + predict_size, spacing[1]*(num_cols-1) + predict_size, self.num_classes))

		for yoffset in range(crop_len, crop_len + spacing[0]*num_lines, spacing[0]):
			for xoffset in range(crop_len, crop_len + spacing[1]*num_cols, spacing[1]):
				input_array = image_array[yoffset - crop_len: yoffset + crop_len, xoffset - crop_len: xoffset + crop_len, :]
				input_array = normalize(input_array, self.image_mean, self.image_std, False)
				input_array = np.expand_dims(input_array, axis=0)

				temp_prob_predict = model.predict_on_batch(input_array)[0]
				temp_predict = self.classifyback(temp_prob_predict)
				temp_predict = to_categorical(temp_predict, self.num_classes).reshape((patch_size, patch_size, self.num_classes)) # one hot representation

				whole_predict[yoffset - crop_len: yoffset - crop_len + predict_size, xoffset - crop_len: xoffset - crop_len + predict_size, :] += \
					temp_predict[offstart: offstart + predict_size, offstart: offstart + predict_size, :]

				num_predict[yoffset - crop_len: yoffset - crop_len + predict_size, xoffset - crop_len: xoffset - crop_len + predict_size] += \
					np.ones((predict_size, predict_size))

				prob_predict[yoffset - crop_len: yoffset - crop_len + predict_size, xoffset - crop_len: xoffset - crop_len + predict_size,:] += \
					np.reshape(temp_prob_predict, (patch_size, patch_size, self.num_classes))[offstart: offstart + predict_size, offstart: offstart + predict_size, :]
			print("Row: " + str(yoffset - crop_len) + ' completed')

		final_predict = self.classifyback(whole_predict)

		prob_predict = prob_predict/np.dstack([num_predict.astype(np.float)]*self.num_classes)

		return final_predict, prob_predict, label_array

	def maskimage(self,
		imagepredict_array: np.ndarray,
		mask_array: np.ndarray,
		mask_dict: Dict[str, str]) -> np.ndarray:
		''' Applies appropriate mask from mask_array to image_array

			imagepredict_array: predicted image array
			mask_array: image to mask image array with
			mask_dict: elements within image_array that can be masked by elements of mask_dict
		'''
		min_labelkey = np.min(list(self.classes.values()))
		min_imagepredict = np.min(imagepredict_array)
		min_maskarray = np.min(mask_array)

		currentimage_array = np.copy(imagepredict_array)

		if min_imagepredict < min_labelkey:
			currentimage_array = currentimage_array + min_labelkey

		if min_maskarray < min_labelkey:
			mask_array = mask_array + min_labelkey

		for k, v in mask_dict.items():
			cover = currentimage_array.astype(int) == self.classes[k] # note that imagepredict will always be from 0 - num_classes
			maskcover = mask_array == self.classes[v]
			currentimage_array[cover & maskcover] = self.classes[v]

		return currentimage_array

	def CRF(self,
		imageprob_array: np.ndarray):
		''' Applies CRF Filter 

			imageprob_array: probability tensor of size H x W x num_classes

			Returns: CRF filtered class image of size H x W
		'''

		min_labelkey = np.min(list(self.classes.values()))
		image_shape = imageprob_array.shape
		ysize, xsize = image_shape[0], image_shape[1] 
		offstart = self.offstart

		prob_predict_switch = np.rollaxis(imageprob_array,2,0)
		U = unary_from_softmax(prob_predict_switch)
		d = dcrf.DenseCRF2D(xsize, ysize, self.num_classes)
		d.setUnaryEnergy(U)
		pairwise_gaussian = create_pairwise_gaussian(sdims=(5,5), shape=image_shape[:2]) # smaller the sdims, the more important it is
		d.addPairwiseEnergy(pairwise_gaussian, compat=0, kernel=dcrf.DIAG_KERNEL, normalization=dcrf.NORMALIZE_SYMMETRIC)
		pairwise_bilateral = create_pairwise_bilateral(sdims=(10,10), schan=3, img = self.image_array[offstart:offstart+ysize, offstart:offstart+xsize], chdim=2) 
		d.addPairwiseEnergy(pairwise_bilateral, compat=10, kernel=dcrf.DIAG_KERNEL, normalization=dcrf.NORMALIZE_SYMMETRIC)

		Q, tmp1, tmp2 = d.startInference()
		for i in range(20):
		    print("KL-divergence at {}: {}".format(i, d.klDivergence(Q)))
		    d.stepInference(Q, tmp1, tmp2)

		MAP = np.argmax(Q, axis=0)
		final_predict = np.reshape(MAP, (ysize, xsize))

		return final_predict + min_labelkey


	def identify_hiprobclasses(self,
		imagepredict_array: np.ndarray,
		imageprob_array: np.ndarray,
		probcutoff_dict: Dict[str, float],
		numcutoff_dict: Dict[str, int] = {},
		targetnum_dict: Dict[str, int] = {},
		n: int = 1000) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
		''' Performs KNN Analysis

			imagepredict_array: image array of classes H x W
			imageprob_array: probability tensor of size H x W x num_classes
			probcutoff_dict: initial lowerbound probability cutoff for classes
			numcutoff_dict: lowerbound numerical cutoff for classes 
			targetnum_dict: target number per classes to hit, will adjuct probcutoff to try to achieve this
			n: number of samples per class
		'''

		offstart = self.offstart # if we only predict on center of an image, then need to know how much to offset

		hiprob_coords = {}
		class_samples = {}

		min_labelkey = np.min(list(self.classes.values()))
		for k, v in probcutoff_dict.items():
			samples = imagepredict_array == self.classes[k]
			samples_cutoff = imageprob_array[:, :, self.classes[k] - min_labelkey] >= v
			samples_hiprob = np.where(samples & samples_cutoff)

			if k in targetnum_dict.keys(): # try to hit aforementioned number of samples, if specified
				probcutoff = v
				while len(samples_hiprob[0]) >= targetnum_dict[k] and probcutoff <= 0.95: # have more samples than required, need to increase cutoff
					probcutoff += 0.01
					samples_hiprob = np.where(imageprob_array[:, :, self.classes[k] - min_labelkey] >= probcutoff)
				while len(samples_hiprob[0]) < targetnum_dict[k] and probcutoff >= 0.5: # have less samples than required, need to decrease cutoff
					probcutoff -= 0.01
					samples_hiprob = np.where(imageprob_array[:, :, self.classes[k] - min_labelkey] >= probcutoff)

			hiprob_coords[k] = samples_hiprob # Location of hi probability samples, if required later
			print("Samples of {} : {}".format(k, len(samples_hiprob[0])))
			randomsample = np.random.randint(len(samples_hiprob[0]), size = n)

			if k in numcutoff_dict.keys(): # if a cutoff is involved for the class
				if len(samples_hiprob[0]) >= numcutoff_dict[k]: # only record if above cutoff threshold
					class_samples[k] = self.image_array[offstart + samples_hiprob[0][randomsample], offstart + samples_hiprob[1][randomsample], :]
			else: # if no cutoff specified, then just record
				class_samples[k] = self.image_array[offstart + samples_hiprob[0][randomsample], offstart + samples_hiprob[1][randomsample], :]

		return hiprob_coords, class_samples

	def KNN_Analysis(self,
		class_samples: Dict[str, np.ndarray],
		KNN_classes: List[str]) -> neighbors.classification.KNeighborsClassifier:
		''' Performs KNN Analysis

			class_samples: Dictionary of arrays of KNN classes
			KNN_classes: All KNN classes to consider (subset of class_samples key values)
		'''
		all_samples = []
		for k, v in class_samples.items():
			if k in KNN_classes:
				print(k, v.shape)
				all_samples.append(v)

		if len(all_samples) >= 2:
			combineALL = np.concatenate((all_samples[0], all_samples[1]), axis = 0)
			y = np.concatenate((np.zeros(all_samples[0].shape[0],), np.ones(all_samples[1].shape[0],)), axis = 0)
			numcombine = 2

			while numcombine < len(all_samples):
				combineALL = np.concatenate((combineALL, all_samples[numcombine]), axis=0)
				y = np.concatenate((y, numcombine*np.ones(all_samples[numcombine].shape[0],)), axis = 0)
				numcombine += 1

		clf_all = neighbors.KNeighborsClassifier(10, weights = 'distance')
		clf_all.fit(combineALL, y)
		return clf_all

	def apply_KNN(self,
		KNN_model: neighbors.classification.KNeighborsClassifier,
		KNN_classes: List[str],
		imagepredict_array: np.ndarray,
		imageprob_array: np.ndarray, 
		reclassify_classes: Dict[str, List[str]]):
		''' Apply the learned KNN filter to image
		'''

		KNN_classdict = {k:c for c,k in enumerate(KNN_classes)}
		min_labelkey = np.min(list(self.classes.values()))

		imgpredict_array = np.copy(imagepredict_array) # create a copy
		prob_array = np.copy(imageprob_array)
		for k, v in reclassify_classes.items():
			class_idx = np.where(imagepredict_array == self.classes[k]) # use the original here, find pixels of certain class
			class_pixels = self.image_array[self.offstart + class_idx[0][:], self.offstart + class_idx[1][:], :] # record RGB + NIR values
			class_predict = KNN_model.predict(class_pixels) # predict on those classes with KNN
			class_prob = KNN_model.predict_proba(class_pixels)

			print(class_pixels.shape, class_predict.shape, class_prob.shape)

			for newclass in v:
				changepixels = np.where(class_predict == KNN_classdict[newclass]) # find predicted pixels that are a certain listed class
				imgpredict_array[class_idx[0][changepixels], class_idx[1][changepixels]] = self.classes[newclass]
				prob_array[class_idx[0][changepixels], class_idx[1][changepixels], :] = 0 
				prob_array[class_idx[0][changepixels], class_idx[1][changepixels], self.classes[newclass] - min_labelkey] \
					= class_prob[changepixels, KNN_classdict[newclass]][0] 

		return imgpredict_array, prob_array


	def cloud_clustering(self,
		cloud_coords: List[np.ndarray],
		distance_multiplier: float,
		dbscan_eps: float,
		dbscan_minsamples: int, 
		repredict_classes_coords: Dict[str, List[np.ndarray]],
		KNN_model: neighbors.classification.KNeighborsClassifier,
		KNN_classes: List[str],
		imagepredict_array: np.ndarray,
		imageprob_array: np.ndarray,
		verbose: bool):

		d = distance_multiplier
		offstart = self.offstart
		imgpredict_array = np.copy(imagepredict_array) # create a copy
		prob_array = np.copy(imageprob_array)
		KNN_classdict = {k:c for c,k in enumerate(KNN_classes)}
		min_labelkey = np.min(list(self.classes.values()))

		coordinates = d * np.stack((offstart + cloud_coords[0], offstart + cloud_coords[1]), axis=-1)
		spectral_and_coords = np.concatenate((self.image_array[offstart + cloud_coords[0], offstart + cloud_coords[1], :], coordinates), axis=-1)
		clustering = DBSCAN(eps=15, min_samples=20).fit(spectral_and_coords)
		uniquelabels = np.unique(clustering.labels_)

		class_mean = {}
		class_std = {}
		for k, v in repredict_classes_coords.items():
			class_mean[k] = np.mean(self.image_array[offstart + v[0], offstart + v[1], :], 0)
			class_std[k] = np.std(self.image_array[offstart + v[0], offstart + v[1], :], 0)

		for l in uniquelabels:
			cluster = np.where(clustering.labels_ == l)
			cloudcluster_mean = np.mean(self.image_array[offstart + cloud_coords[0][cluster], offstart + cloud_coords[1][cluster]], 0)
			cloudcluster_std = np.std(self.image_array[offstart + cloud_coords[0][cluster], offstart + cloud_coords[1][cluster]], 0)

			if verbose:
				print("Cloud cluster {} mean/std: {}, {}".format(l, cloudcluster_mean, cloudcluster_std))

			for k in repredict_classes_coords.keys(): # check if cloud cluster is within other cluster (usually coral or sediment)
				if all((cloudcluster_mean >= class_mean[k] - 2*class_std[k]) & (cloudcluster_mean <= class_mean[k] + 2*class_std[k])) and l >= 0:
					# do reprediction
					clouds_pixels = self.image_array[offstart + cloud_coords[0][cluster], offstart + cloud_coords[1][cluster], :]
					clouds_predict = KNN_model.predict(clouds_pixels) # predict on those classes with KNN
					clouds_prob = KNN_model.predict_proba(clouds_pixels)
					
					for c in KNN_classes:
						changepixels = np.where(clouds_predict == KNN_classdict[c]) # find predicted pixels that are a certain listed class

						imgpredict_array[cloud_coords[0][cluster][changepixels], cloud_coords[1][cluster][changepixels]] = self.classes[c]
						prob_array[cloud_coords[0][cluster][changepixels], cloud_coords[1][cluster][changepixels], :] = 0 
					
						prob_array[cloud_coords[0][cluster][changepixels], cloud_coords[1][cluster][changepixels], self.classes[c] - min_labelkey] \
							= clouds_prob[changepixels, KNN_classdict[c]][0] 

					continue

		return imgpredict_array, prob_array


	@staticmethod
	def classifyback(predictions: np.ndarray) -> np.ndarray: 
		''' Classify from categorical array to label
			# Input:
			# 	predictions: Array of vectorized categorical predictions, nrow x ncol x n_categories
			# Output:
			# 	Array of label predictions, nrow x ncol
		'''
		return np.argmax(predictions,-1)


