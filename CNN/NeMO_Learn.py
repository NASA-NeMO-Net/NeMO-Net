import os
import yaml
import sys
import datetime
import numpy as np
import json
import keras.backend as K
import tensorflow as tf

from matplotlib import pyplot as plt

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
# import NeMO_layers
# from keras.models import Model, Sequential, load_model
# from keras.callbacks import (
#     ReduceLROnPlateau,
#     CSVLogger,
#     EarlyStopping,
#     ModelCheckpoint,
#     TerminateOnNaN)
# from NeMO_callbacks import CheckNumericsOps, WeightsSaver
from NeMO_Generator import NeMOImageGenerator, ImageSetLoader
from NeMO_DirectoryIterator import NeMODirectoryIterator
from NeMO_Models import NeMO_FCN

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

	conv_layers = 5
	full_layers = 0

	# First 4 megablocks of the resnet-50 architecture

	conv_params = {"filters": [[64] , [([64,64,128],128)]*3, [([128,128,256],256)]*4, [([256,256,512],512)]*6, [([512,512,1024],1024)]*3],
		"conv_size": [[(7,7)] , [([(1,1),(3,3),(1,1)], (1,1))]*3, [([(1,1),(3,3),(1,1)], (1,1))]*4, [([(1,1),(3,3),(1,1)], (1,1))]*6, [([(1,1),(3,3),(1,1)], (1,1))]*3],
		"conv_strides": [(2,2), [([(1,1),(1,1),(1,1)], (1,1))] + [(1,1)]*2 , [([(2,2),(1,1),(1,1)], (2,2))] + [(1,1)]*3 , [([(2,2),(1,1),(1,1)], (2,2))] + [(1,1)]*5, [([(2,2),(1,1),(1,1)], (2,2))] + [(1,1)]*2],
		"padding": ['same', 'same', 'same', 'same', 'same'],
		"dilation_rate": [(1,1), (1,1), (1,1), (1,1), (1,1)],
		"pool_size": [(3,3), (1,1), (1,1), (1,1), (1,1)],
		"pool_strides": [(2,2), (1,1), (1,1), (1,1), (1,1)],
		"pad_size": [(0,0), (0,0), (0,0), (0,0), (0,0)],
		"filters_up": [None]*conv_layers,
		"upconv_size": [None]*conv_layers,
		"upconv_strides": [None]*conv_layers,
		"layercombo": ["cbap", [("cbacbac","c")]+[("bacbacbac","")]*2, [("bacbacbac","c")]+[("bacbacbac","")]*3, [("bacbacbac","c")]+[("bacbacbac","")]*5, [("bacbacbac","c")]+[("bacbacbac","")]*2], 
		"layercombine": ["","sum","sum","sum","sum"],           
		"full_filters": [1024,1024], 
		"dropout": [0,0]}

	RCU = ("bacbac","")
	CRPx2 = ([("pbc",""),"pbc"],"")

	bridge_params = {"filters": [[1024,1024,128], [512,512,64], [256,256,32], [128,128,16]],
		"conv_size": [(3,3), (3,3), (3,3), (3,3)],
		"filters_up": [None]*4,
		"upconv_size": [None]*4,
		"upconv_strides": [None]*4,
		"layercombo": [[RCU,RCU,"bc"], [RCU,RCU,"bc"], [RCU,RCU,"bc"], [RCU,RCU,"bc"]],
		"layercombine": ["sum","sum","sum","sum"]}

	prev_params = {"filters": [None, [128,128,64], [64,64,32], [32,32,16]],
		"conv_size": [None, (3,3),(3,3),(3,3)],
		"filters_up": [None,None,None,None],
		"upconv_size": [None,None,None,None],
		"upconv_strides": [None,(2,2),(2,2),(2,2)],
		"upconv_type": [None,"bilinear","bilinear","bilinear"],
		"layercombo": ["", [RCU,RCU,"bcu"], [RCU,RCU,"bcu"], [RCU,RCU,"bcu"]],
		"layercombine": [None,"sum","sum","sum"]} 

	next_params = {"filters": [["",128,128], ["",64,64], ["",32,32], ["",16,16,16,None,16,None]],
		"conv_size": [(3,3), (3,3), (3,3), (3,3)],
		"pool_size": [(5,5), (5,5), (5,5), (5,5)],
		"filters_up": [None,None,None,None],
		"upconv_size": [None,None,None,None],
		"upconv_strides": [None,None,None,(2,2)],
		"upconv_type": [None,None,None,"bilinear"],
		"layercombo": [["a",CRPx2,RCU], ["a",CRPx2,RCU], ["a",CRPx2,RCU], ["a",CRPx2,RCU,RCU,"u",RCU,"u"]],
		"layercombine": ["sum","sum","sum","sum"]} 

	decoder_index = [0,1,2,3]
	# upsample = [False,True,True,True,True]
	scales= [1,1,1,1]

	RefineMask = NeMO_FCN(input_shape=(y,x,num_channels), 
		classes=num_classes, 
		decoder_index = decoder_index, 
		weight_decay=3e-3, 
		trainable_encoder=True,
		conv_blocks=conv_layers, 
		dense_blocks=full_layers, 
		conv_params=conv_params, 
		scales=scales, 
		bridge_params=bridge_params, 
		prev_params=prev_params, 
		next_params=next_params, 
		reshape=False) # default reshape = True

	
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