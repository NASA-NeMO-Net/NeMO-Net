import os
import yaml
import sys
import datetime
import numpy as np
import json

from matplotlib import pyplot as plt

import tensorflow as tf
import keras.backend as K
from keras.optimizers import Adam
from keras.utils.layer_utils import print_summary
from keras.callbacks import (
	ReduceLROnPlateau,
	CSVLogger,
	EarlyStopping,
	ModelCheckpoint,
	TerminateOnNaN)


# Set up GPU
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
global _SESSION
config = tf.ConfigProto(allow_soft_placement=True)
config.gpu_options.allow_growth = True
_SESSION = tf.Session(config=config)
K.set_session(_SESSION)

# NeMO-Net specific files 
sys.path.append("./utils/") # Adds higher directory to python modules path.
from NeMO_CoralData import CoralData
# from NeMO_models import AlexNetLike, SharpMask_FCN
# from NeMO_generator import NeMOImageGenerator, ImageSetLoader
# from NeMO_backend import get_model_memory_usage
# from NeMO_losses import charbonnierLoss, keras_lovasz_softmax, categorical_focal_loss

# from NeMO_callbacks import CheckNumericsOps, WeightsSaver
from NeMO_Generator import NeMOImageGenerator, ImageSetLoader
from NeMO_DirectoryIterator import NeMODirectoryIterator
from NeMO_Architecture import NeMO_FCN_Architecture
from NeMO_Models import NeMO_FCN
from NeMO_Backend import get_model_memory_usage

def run_training() -> None:
	CoralClasses_config = './config/CoralClasses.json'
	with open(CoralClasses_config) as json_file:
		CoralClasses = json.load(json_file)
	labelkey = CoralClasses["VedConsolidated_ClassDict"]
	num_classes = len(labelkey)

	DataSource_config = "./config/RefineNet_DataSource.yaml" # Training files info
	with open(DataSource_config, 'r') as stream:
		try:	
			init_args = yaml.safe_load(stream)
		except yaml.YAMLError as exc:
			print(exc)

	train_loader = ImageSetLoader(**init_args['image_set_loader']['train'])
	val_loader = ImageSetLoader(**init_args['image_set_loader']['val'])
	y = train_loader.image_size[1]
	x = train_loader.image_size[0]
	num_channels = 4
	batch_size = 8

	NeMO_ImageGenerator = NeMOImageGenerator(image_shape=[y, x, num_channels],
		pixel_mean = None,
		pixel_std = None,
		channel_shift_range = 0,
		spectral_augmentation = True,
		random_rotation = True)

	DirectoryIterator_train = NeMO_ImageGenerator.flow_from_NeMOdirectory(directory = train_loader.image_dir,
		label_directory = train_loader.label_dir,
		classes = labelkey,
		class_weights = None,
		image_size = (y,x), 
		color_mode = '8channel_to_4channel',
		batch_size = batch_size,
		shuffle = True,
		seed = None,
		save_to_dir = None,
		save_prefix = "test",
		save_format = "png",
		reshape = True)

	DirectoryIterator_valid = NeMO_ImageGenerator.flow_from_NeMOdirectory(directory = val_loader.image_dir,
		label_directory = val_loader.label_dir,
		classes = labelkey,
		class_weights = None,
		image_size = (y,x), 
		color_mode = '8channel_to_4channel',
		batch_size = batch_size,
		shuffle = True,
		seed = None,
		save_to_dir = None,
		save_prefix = "test",
		save_format = "png",
		reshape = True)


	FCN_architecture = NeMO_FCN_Architecture(conv_blocks = 5, dense_blocks = 0, decoder_index = [0,1,2,3], scales = [1,1,1,1])
	FCN_architecture.refinemask_defaultparams()

	RefineMask = NeMO_FCN(input_shape = (y,x,num_channels), 
		classes = num_classes, 
		architecture = FCN_architecture,
		weight_decay = 3e-3, 
		trainable_encoder = True,
		reshape = True) # default reshape = True

	optimizer = Adam(1e-4)
	print_summary(RefineMask, line_length=150, positions=[.35, .55, .65, 1.])

	RefineMask.compile(optimizer=optimizer, loss = 'categorical_crossentropy', metrics=['accuracy'])

	print("Memory required (GB): ", get_model_memory_usage(batch_size, RefineMask))

	RefineMask.fit_generator(DirectoryIterator_train,
		steps_per_epoch=10,
		epochs=1,
		validation_data=DirectoryIterator_valid,
		validation_steps=10,
		verbose=1)
	#     callbacks=[lr_reducer, early_stopper, nan_terminator, checkpointer])

	
	# imgpath = '/home/shared/NeMO-Net Data/processed_Fiji_files/mosaiced_003.TIF'
	# labelpath = '/home/shared/NeMO-Net Data/KSLOF-CICIA-FIJI/Cicia_Habitat_classes_raster_KSLOF.tif'
	
	# Fiji = CoralData(imagepath = imgpath, labelpath = labelpath, labelkey = CoralClasses["Fiji_ClassDict"], load_type = "raster", shpfile_classname = 'Hab_name')
	# Fiji.consolidate_classes(newclassdict = CoralClasses["L3_ClassDict"], transferdict = CoralClasses["Fiji2L3_Dict"])
	# Fiji.consolidate_classes(newclassdict = CoralClasses["VedConsolidated_ClassDict"], transferdict = CoralClasses["L32VedConsolidated_Dict"])

	# Fiji.export_trainingset(exporttrainpath = '../Images/TestNewCoralUtils/images/', 
	# 	exportlabelpath = '../Images/TestNewCoralUtils/labels/', 
	# 	txtfilename = 'test_txtfile.txt', 
	# 	image_size = 256, 
	# 	N = 1, 
	# 	magnification = 1.0, 
	# 	magimg_path = None,
	# 	subdir = True, 
	# 	cont = False, 
	# 	consolidated = True, 
	# 	classestoexport = ['Coral'], 
	# 	mosaic_mean = 0.0, 
	# 	mosaic_std = 1.0, 
	# 	thresholds = [0, 255],
	# 	bandstoexport = [5, 3, 2], 
	# 	label_cmap = None)

	# model = load_model('./tmp/RefineMask_Jarrett256_RGB_NIR_spectralshift.h5', custom_objects={'BilinearUpSampling2D':NeMO_layers.BilinearUpSampling2D, 'charbonnierLoss': charbonnierLoss})
	
	# ystart = 1186-256
	# xstart = 4105-256
	# ylen = 512
	# xlen = 512
	# image_array = Fiji.image[ystart: ystart+ylen, xstart:xstart+xlen, :]
	# image_array = np.delete(image_array, [0,3,5,7], 2)
	# num_classes = len(CoralClasses["VedConsolidated_ClassDict"])

	# final_predict, prob_predict = Fiji.predict_on_whole_image(model = model, 
	# 	image_array = image_array,
	# 	num_classes = num_classes,
	# 	image_mean = 100.0,
	# 	image_std = 100.0, 
	# 	patch_size = 256,
	# 	num_lines = None, 
	# 	spacing = (128, 128), 
	# 	predict_size = 128)

if __name__ == "__main__":
	run_training()