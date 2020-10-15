import os
import yaml
import sys
import datetime
import numpy as np
import json
import keras.backend as K
import tensorflow as tf

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
from NeMO_models import AlexNetLike, SharpMask_FCN
from NeMO_generator import NeMOImageGenerator, ImageSetLoader
from NeMO_backend import get_model_memory_usage
from NeMO_losses import charbonnierLoss, keras_lovasz_softmax, categorical_focal_loss
import NeMO_layers
from keras.models import load_model
from keras.callbacks import (
    ReduceLROnPlateau,
    CSVLogger,
    EarlyStopping,
    ModelCheckpoint,
    TerminateOnNaN)
from NeMO_callbacks import CheckNumericsOps, WeightsSaver
from NeMO_Generator import NeMOImageGenerator
from NeMO_DirectoryIterator import NeMODirectoryIterator

def run_training() -> None:
	CoralClasses_config = './config/CoralClasses.json'
	with open(CoralClasses_config) as json_file:
	    CoralClasses = json.load(json_file)
	labelkey = CoralClasses["VedConsolidated_ClassDict"]

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

	imgpath = '/home/shared/NeMO-Net Data/processed_Fiji_files/mosaiced_003.TIF'
	labelpath = '/home/shared/NeMO-Net Data/KSLOF-CICIA-FIJI/Cicia_Habitat_classes_raster_KSLOF.tif'

	Fiji = CoralData(imagepath = imgpath, labelpath = labelpath, labelkey = CoralClasses["Fiji_ClassDict"], load_type = "raster")
	Fiji.consolidate_classes(newclassdict = CoralClasses["L3_ClassDict"], transferdict = CoralClasses["Fiji2L3_Dict"])
	Fiji.consolidate_classes(newclassdict = CoralClasses["VedConsolidated_ClassDict"], transferdict = CoralClasses["L32VedConsolidated_Dict"])

	Fiji.export_trainingset(exporttrainpath = '../Images/TestNewCoralUtils/images/', 
		exportlabelpath = '../Images/TestNewCoralUtils/labels/', 
		txtfilename = 'test_txtfile.txt', 
		image_size = 256, 
		N = 1, 
		magnification = 1.0, 
		magimg_path = None,
		subdir = True, 
		cont = False, 
		consolidated = True, 
		classestoexport = ['Coral'], 
		mosaic_mean = 0.0, 
		mosaic_std = 1.0, 
		thresholds = [0, 255],
		bandstoexport = [5, 3, 2], 
		label_cmap = None)
	
	# NeMO_ImageGenerator = NeMOImageGenerator(image_shape=[y, x, num_channels],
 #                                    pixel_mean = None,
 #                                    pixel_std = None,
 #                                    channel_shift_range = 0,
 #                                    spectral_augmentation = True,
 #                                    random_rotation = True)


	# DirectoryIterator = NeMO_ImageGenerator.flow_from_NeMOdirectory(directory = train_loader.image_dir,
	# 	label_directory = train_loader.label_dir,
	# 	classes = labelkey,
	# 	class_weights = None,
	# 	image_size = (y,x), 
	# 	color_mode = '8channel_to_4channel',
	# 	batch_size = 1,
	# 	shuffle = True,
	# 	seed = None,
	# 	save_to_dir = None,
	# 	save_prefix = "test",
	# 	save_format = "png",
	# 	reshape = True)

	# batch_x, batch_y = DirectoryIterator.next()

if __name__ == "__main__":
	run_training()