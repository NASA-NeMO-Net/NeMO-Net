import os
import yaml
import sys
import datetime
import numpy as np
import json

import keras.backend as K
from keras.models import load_model
import tensorflow as tf
from matplotlib import pyplot as plt

# NeMO-Net specific files 
sys.path.append("./utils/") # Adds higher directory to python modules path.
import NeMO_Layers
from NeMO_Generator import NeMOImageGenerator, ImageSetLoader
from NeMO_CoralData import CoralData
from NeMO_Generator import NeMOImageGenerator
from NeMO_DirectoryIterator import NeMODirectoryIterator
from NeMO_CoralPredict import CoralPredict
from NeMO_losses import charbonnierLoss, keras_lovasz_softmax, categorical_focal_loss

import tensorflow as tf
from tensorflow.python.client import device_lib
os.environ["CUDA_VISIBLE_DEVICES"] = "2"

num_cores = 4
num_GPU = 1
num_CPU = 1

global _SESSION
config = tf.ConfigProto(intra_op_parallelism_threads=num_cores,
	inter_op_parallelism_threads=num_cores, allow_soft_placement=True,
	device_count = {'CPU' : num_CPU, 'GPU' : num_GPU})
_SESSION = tf.Session(config=config)
K.set_session(_SESSION)

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

Fiji = CoralData(imagepath = imgpath, labelpath = labelpath, labelkey = CoralClasses["Fiji_ClassDict"], load_type = "raster", shpfile_classname = 'Hab_name')
Fiji.consolidate_classes(newclassdict = CoralClasses["L3_ClassDict"], transferdict = CoralClasses["Fiji2L3_Dict"])
Fiji.consolidate_classes(newclassdict = CoralClasses["VedConsolidated_ClassDict"], transferdict = CoralClasses["L32VedConsolidated_Dict"])

model = load_model('./tmp/RefineMask_Jarrett256_RGB_NIR_spectralshift.h5', custom_objects={'BilinearUpSampling2D':NeMO_Layers.BilinearUpSampling2D, 'charbonnierLoss': charbonnierLoss})

ystart = 1186-256
xstart = 4105-256
ylen = 512
xlen = 512
image_array = Fiji.image[ystart: ystart+ylen, xstart:xstart+xlen, :]
image_array = np.delete(image_array, [0,3,5,7], 2)

label_array = np.copy(Fiji.labelimage_consolidated[ystart: ystart+ylen, xstart:xstart+xlen])

NeMOPredict = CoralPredict(image_array, 
	labelkey, 
	label_array = label_array, 
	patch_size = 256,
	spacing = (128, 128),
	predict_size = 128,
	image_mean=100.0, 
	image_std = 100.0)
final_predict, prob_predict, label_array = NeMOPredict.predict_on_whole_image(model = model, 
	num_lines = None)

mask_dict = {'Coral': 'Terrestrial vegetation',
	'Sediment': 'Terrestrial vegetation',
	'Beach': 'Terrestrial vegetation',
	'Seagrass': 'Terrestrial vegetation',
	'Deep water': 'Terrestrial vegetation',
	'Wave breaking': 'Terrestrial vegetation'}
final_predict = NeMOPredict.maskimage(final_predict, label_array, mask_dict)
CRF_image = NeMOPredict.CRF(imageprob_array = prob_predict)

n = 1000
probcutoff_dict = {'Sediment': 0.90, 'Coral': 0.85, 'Seagrass': 0.75}
KNN_classes = ['Sediment', 'Coral', 'Seagrass']
numcutoff_dict = {'Seagrass': 300}
targetnum_dict = {'Coral': 1000}

clf = NeMOPredict.KNN_Analysis(CRF_image, 
	prob_predict, 
	probcutoff_dict,
	KNN_classes = KNN_classes,
	numcutoff_dict = numcutoff_dict,
	targetnum_dict = targetnum_dict,
	n=1000)

reclassify_classes = {'Coral': ['Sediment', 'Coral', 'Seagrass']}
final_predict, finalprob_predict = NeMOPredict.apply_KNN(clf,
	CRF_image,
	prob_predict,
	reclassify_classes)