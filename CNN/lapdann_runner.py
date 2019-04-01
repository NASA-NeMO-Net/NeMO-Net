import os
import time
import numpy as np
from glob import glob
import tensorflow as tf
from lapdann import LAPDANN

print("Libraries are loaded.")

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
gpu_options = tf.GPUOptions(allow_growth=True)				# per_process_gpu_memory_fraction=0.4)
print("GPU allocations are set.")


island = "PerosBanhos" # "Cicia"

def to_categorical(y, num_classes=None, dtype='float32'):
	y = np.array(y, dtype='int')
	input_shape = y.shape
	if input_shape and input_shape[-1] == 1 and len(input_shape) > 1:
		input_shape = tuple(input_shape[:-1])
	y = y.ravel()
	if not num_classes:
		num_classes = np.max(y) + 1
	n = y.shape[0]
	categorical = np.zeros((n, num_classes), dtype=dtype)
	categorical[np.arange(n), y] = 1
	output_shape = input_shape + (num_classes,)
	categorical = np.reshape(categorical, output_shape)
	return categorical


# Import the Coral dataset with labels
# Training
trainDX, trainX, trainY = np.zeros((0, 80, 80, 4)), np.zeros((0, 400, 400, 4)), np.zeros((0, 400, 400, 1))
#files = glob("../Data/train_*0.npy")[:-1]	# Scenario III
#files = glob("../Data/train_*0.npy") 		# Scenario II
files = glob("../Data/train_Peros*")[:1]	# Scenario I
findex = np.random.randint(len(files), size=len(files))

with open("selected_training_file_list.txt", "w") as sfiles:
	for ind in findex:
		sfiles.write(files[ind])
for find in findex:
	mat = np.load(files[find]).item()
	if "trainX" in mat.keys():
		trainX = np.concatenate((trainX, mat["trainX"]))
		trainDX = np.concatenate((trainDX, mat["trainDX"]))
		trainY = np.concatenate((trainY, mat["trainY"][..., np.newaxis]))
	elif "dataX" in mat.keys():
		trainX = np.concatenate((trainX, mat["dataX"]))
		trainDX = np.concatenate((trainDX, mat["dataDX"]))
		trainY = np.concatenate((trainY, mat["dataY"]))
mat = None
w, h, d = trainY.shape[1], trainY.shape[2], len(np.unique(trainY))
trainY = to_categorical(trainY.reshape(-1, 1) - 1.0, num_classes=d).reshape(-1, w, h, d)

# Testing
mat = np.load("../Data/test_{}_0.npy".format(island)).item()
if "testX" in mat.keys():
	testX, testY, testDX = mat["testX"][:1000], mat["testY"][..., np.newaxis][:1000], mat["testDX"][:1000]
elif "dataX" in mat.keys():
	testX, testY, testDX = mat["dataX"][:1000], mat["dataY"][:1000], mat["dataDX"][:1000]
mat = None
testY = to_categorical(testY.reshape(-1, 1) - 1.0, num_classes=d).reshape(-1, w, h, d)

# Rescale -1 to 1
pixel_mean = (trainX.mean((0, 1, 2)) + trainDX.mean((0, 1, 2))) / 2.
trainX = (trainX.astype(np.float32) - pixel_mean) / 255.
trainDX = (trainDX.astype(np.float32) - pixel_mean) / 255.
testX = (testX.astype(np.float32) - pixel_mean) / 255.
testDX = (testDX.astype(np.float32) - pixel_mean) / 255.
print("Loading data is done.")

print("Training...")
t = time.time()
model =  LAPDANN(width=80, height=80, channels=4, classes=10, batch_size=32, scale_list=[80, 200, 400])
with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
	sess.run(tf.global_variables_initializer())
#	model.train(sess, trainX, trainDX, trainY, epochs=50, shuffle=False, verbose=True)
#	model.save_model(sess, filepath="../Models/LAPDANN_01")
#	model.load_model(sess, filepath="../Models/LAPDANN_01")

	model.load_lapgan(sess, filepath="../Models/LAPDANN_01")
#	model.train_dann_only(sess, trainX, trainDX, trainY, epochs=50, shuffle=False, verbose=True)
	model.save_dann(sess, filepath="../Models/LAPDANN_01")

	predDX, preddannDX = model.test(sess, testDX, batch_size=8)

	predY0, predY1, predY2 = model.test_lapgan(sess, testY, batch_size=8)
print("The training took {} seconds.".format(time.time()-t))


np.save("../Results/{}_LAPDANN_predDX.npy".format(island), predDX)
np.save("../Results/{}_LAPDANN_preddannDX.npy".format(island), preddannDX)

np.save("../Results/{}_LAPDANN_testY.npy".format(island), testY)

np.save("../Results/{}_LAPDANN_predY0.npy".format(island), predY0)
np.save("../Results/{}_LAPDANN_predY1.npy".format(island), predY1)
np.save("../Results/{}_LAPDANN_predY2.npy".format(island), predY2)
