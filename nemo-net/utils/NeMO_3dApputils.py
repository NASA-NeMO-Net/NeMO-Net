import sys
sys.path.append("./utils/") # Adds higher directory to python modules path.
import random
import numpy as np
import cv2
import yaml
import glob, os
import json
import importlib
import itertools
import collections
import loadcoraldata_utils as coralutils
from typing import List, NamedTuple
import pandas as pd
import loadcoraldata_utils as coralutils

from matplotlib import colors

class CoralImage(NamedTuple):
	# Contains data regarding a single image
	image: np.ndarray
	classification: np.ndarray
	user_ID: str
	transect_name: str
	classification_ID: str
	rating: float
	perc_classified: float
	num_classes: int

class CoralImageList(collections.MutableSequence):
	def __init__(self, *args):
		self.list = list()
		self.extend(list(args))

	def check(self, v):
		if not isinstance(v, CoralImage):
			raise (TypeError, v)

	def __len__(self): return len(self.list)

	def __getitem__(self, i): return self.list[i]

	def __delitem__(self, i): del self.list[i]

	def __setitem__(self, i, v):
		self.check(v)
		self.list[i] = v

	def insert(self, i, v):
		self.check(v)
		self.list.insert(i, v)

	def prune_perc_classified(self, perc_classified: float):
		for item in self.list:
			if item.perc_classified < perc_classified:
				self.list.remove(item)

	def output_training(self, directory: str, add_to_dir: bool = False):
		trainpath = os.path.join(directory, 'TrainImages')
		labelpath = os.path.join(directory, 'LabelImages')

		if not os.path.exists(trainpath):
			os.mkdir(trainpath)
		if not os.path.exists(labelpath):
			os.mkdir(labelpath)

		trainfiles = os.listdir(trainpath)
		labelfiles = os.listdir(labelpath)
		if add_to_dir:
			numfiles = len(trainfiles)
		else:
			for f in trainfiles:
				os.remove(os.path.join(trainpath, f))
			for f in labelfiles:
				os.remove(os.path.join(labelpath, f))
			numfiles = 0

		for CI in self.list:
			if CI.image.shape[2] == 3: # RGB data 
				cv2.imwrite(os.path.join(trainpath, 'Coral_' + str(numfiles).zfill(8)) + '.png', cv2.cvtColor(CI.image, cv2.COLOR_BGR2RGB))
			elif CI.image.shape[2] == 8: # WV2 8-channel data
				coralutils.CoralData.export_raster(CI.image, os.path.join(trainpath, 'Coral_' + str(numfiles).zfill(8)) + '.tif')
			cv2.imwrite(os.path.join(labelpath, 'Coral_' + str(numfiles).zfill(8)) + '.png', np.uint8(CI.classification*255/CI.num_classes))
			numfiles += 1

	def __str__(self):
		return str(self.list)

class Coral3DData:

	# Load from master csv file: Make sure the following columns exist:
	# ["ClassificationID", "Mesh", "Classification", "Rating", "ZoneName", "UserID", "UserEmail", "UserEXP", "UserHoursPlayed"]
	def __init__(self, masterfile, img_path=None ,meshType = None, n_files = None) :
	# masterFile: Master .csv file to load
	# meshType: ["satellite", "2D", "<Mesh.fbx>"], None will load all types
	# img_path: Path where all user classifications are stored (organized by folders)
	# n_files: # of entries to load, None will load all entries
		self.columns = ["Date_Created", "Transect_fileID", "Classification_fileID", "Rating", "Zone", "User_ID", "User_Email", "User_EXP", "Hours_Played"]
		self.img_path = img_path
		count = 0

		with open(masterfile, 'r') as f:
		    for line in f:
		        count += 1
		print("Total number of lines in " + masterfile + " is:", count)

		master_meshType = ["2D", "satellite", "3D"]
		if meshType is None:
			print("Will attempt to load all entries (2D, satellite, and 3D)...")
			self.meshType = master_meshType

		# Check meshType is list
		if not isinstance(meshType, list):
			print("Input mesh type is not a list!")
			raise TypeError

		# Check meshtypes (2D, satellite, 3D)
		for mesh in meshType:
			if mesh not in master_meshType:
				print("Check meshType list, incorrect mesh types specified! ['2D', 'satellite', '3D']")
				raise ValueError

		print("Will attempt to load only", meshType, "data")
		self.meshType = meshType

		if n_files is None:
			self.n_files = count - 1
			print("Will attempt to load all relevant files...")
		else:
			self.n_files = n_files
			print("Will attempt to load only", n_files, "relevant files...")
		count = 0
		
		self.NeMO_df = pd.read_csv(masterfile, sep=',', header=0)

		# some cleaning up of loaded dataframe
		if "3D" in self.meshType:
			idx = (self.NeMO_df.Mesh.isin(self.meshType)) | (~self.NeMO_df.Mesh.str.match("2D") & ~self.NeMO_df.Mesh.str.match("satellite") & self.NeMO_df.Mesh.str.contains("fbx"))
		else:
			idx = (self.NeMO_df.Mesh.isin(self.meshType)) | (~self.NeMO_df.Mesh.str.match("2D") & ~self.NeMO_df.Mesh.str.match("satellite") & ~self.NeMO_df.Mesh.str.contains("fbx"))
		self.NeMO_df = self.NeMO_df[idx]

		self.NeMO_df.rename(columns={"DateCreated": "Date_Created",
			"Mesh": "Image_Type",
			"Classification": "Classification_fileID",
			"Rating": "Rating",
			"ZoneName": "Zone",
			"UserID": "User_ID",
			"UserEmail": "User_Email",
			"UserEXP": "User_EXP",
			"UserHoursPlayed": "Hours_Played"}, inplace=True)

		self.NeMO_df["Date_Created"] = self.NeMO_df["Date_Created"].apply(lambda x: pd.to_datetime(x))
		self.NeMO_df["Image_Type"] = self.NeMO_df["Image_Type"].apply(lambda x: "3D" if ("fbx" in x) else x)
		self.NeMO_df['Classification_fileID'] = self.NeMO_df["Classification_fileID"].apply(lambda x: x.split("/")[-1])

		def extract_transect_fileID(x):
			out = x.split("/")[-1]
			out = out.replace("+", " ")
			out = out[:-4]
			if "modified" in out:
				out = out[:-9]
			return out
		transect_fileID = self.NeMO_df["Texture"].apply(extract_transect_fileID)
		self.NeMO_df.insert(0, "Transect_fileID", transect_fileID)

		transect_path = self.NeMO_df["Texture"].apply(lambda x: x.split("/")[3:-1])
		self.NeMO_df.insert(1, "Transect_Path", transect_path)

		image_format = self.NeMO_df["Texture"].apply(lambda x: x[-3:])
		self.NeMO_df.insert(2, "Image_Format", image_format)

		self.NeMO_df.drop(columns=["ClassificationID", "Texture"], inplace=True)

		self.unique_transects = self.NeMO_df.Transect_fileID.unique()
		self.set_colorclass_dict()

	def load_satellite_transect_data(self): # this is fixed
		Alan_Transect_file = "/home/rechant/Documents/NeMO-Net/NeMO-NET/Images/AppData_May/NeMO-MayCSVs/CoralInfoAlan.txt"
		self.satellite_df = pd.read_csv(Alan_Transect_file, " ", header=None)
		self.satellite_df[1] = self.satellite_df[1].apply(lambda x: x.split("/")[-1])
		transect_IDList = ["Coral_AT_" + str(i).zfill(8) + ".png" for i in range(len(self.satellite_df))]
		self.satellite_df.insert(1, "Transect_fileID", transect_IDList)

		self.satellite_df.drop(columns=[0], inplace=True)
		self.satellite_df.rename(columns={1: "Mosaic_ID",
			2: "ystart",
			3: "xstart"}, inplace=True)

		Satellite_Transect_file = "/home/rechant/Documents/NeMO-Net/NeMO-NET/Images/AppData_May/NeMO-MayCSVs/CoralInfo.txt"
		satellite_df2 = pd.read_csv(Satellite_Transect_file, " ", header=None)
		satellite_df2[0] = satellite_df2[0].apply(lambda x: x.split("/")[-1])
		satellite_df2.rename(columns={0: "Transect_fileID",
			1: "Mosaic_ID",
			2: "ystart",
			3: "xstart"}, inplace=True)

		self.satellite_df = pd.concat([self.satellite_df, satellite_df2])
		
	def set_colorclass_dict(self):
		# default coral colors
		self.coralcolors3D_dict = { "No Data": "#000000",
		       "OutofBounds": "#334C4C",
		       "Unknown": "#33B2CC",
		       "Bare Substratum": "#F38484",
		       "Acroporidae": "#9AEFAB",
		       "Poritidae": "#EAEF70",
		       "Gorgoniidae": "#F885EB",
		       "Merulinidae": "#85F7F8",
		       "Montastraeidae": "#AD77EC",
		       "Mussidae": "#DF981C",
		       "Invertebrate": "#CB0C0C",
		       "Agariciidae": "#9F3333",
		       "Siderastreidae": "#710067",
		       "Pocilloporidae": "#2C50B4",
		       "Alcyoniidae": "#7EA4AC",
		       "Fungiidae": "#D0B987",
		       "Green Algae": "#00FF66",
		       "Brown Algae": "#A87700",
		       "Red Algae": "#FF00A2",
		       "Seagrass": "#69B300",
		       "Mangrove": "#6d3E8E",
		       "Plexauridae": "#FF8400"}

		self.alternateColors_dict = {"Acroporidae": np.asarray([103,230,129],dtype=np.uint8)}

		self.satellitecolors_dict = { "Coral": "#FF81C0",
			"Sediment" : "#720068",
			"Beach": "#D1B26f", 
			"Seagrass": "#69B300",
			"Terrestrial Vegetation": "#FF1300",
			"Deep Water": "#0343DF",
			"Clouds": '#FFFF00',
			"Wave Breaking": "#F97306",
			"Other or Unknown": "#33B2CC"}

		self.cmap3D = [self.coralcolors3D_dict[k] for k in self.coralcolors3D_dict.keys()]
		self.cmap3D = colors.ListedColormap(self.cmap3D)
		self.bounds3D = [i-0.5 for i in range(self.cmap3D.N + 1)]
		self.norm3D = colors.BoundaryNorm([b+0.5 for b in self.bounds3D], self.cmap3D.N)
		self.labelkey3D = [np.uint8(255/self.cmap3D.N*i) for i in range(self.cmap3D.N)] # Assuming labels are saved according to # of consolclass

		self.coral3D_dict = {}
		counter = 0
		for k in self.coralcolors3D_dict.keys():
			self.coral3D_dict[k] = counter
			counter += 1

		self.cmap_sat = [self.satellitecolors_dict[k] for k in self.satellitecolors_dict.keys()]
		self.cmap_sat = colors.ListedColormap(self.cmap_sat)
		self.bounds_sat = [i-0.5 for i in range(self.cmap_sat.N + 1)]
		self.norm_sat = colors.BoundaryNorm([b+0.5 for b in self.bounds_sat], self.cmap_sat.N)
		self.labelkey_sat = [np.uint8(255/self.cmap_sat.N*i) for i in range(self.cmap_sat.N)] # Assuming labels are saved according to # of consolclass

		self.coralsat_dict = {}
		counter = 0
		for k in self.satellitecolors_dict.keys():
			self.coralsat_dict[k] = counter
			counter += 1

	def load_transect_file(self, classification_fileID: str) -> np.ndarray:
		# Try to find classification image record in loaded pd dataframe
		selected_pd_item = self.NeMO_df[self.NeMO_df['Classification_fileID'] == classification_fileID]
		if len(selected_pd_item) == 0:
			print(classification_fileID + " not found in loaded pandas dataframe... cannot load file!")
			raise FileNotFoundError
		else:
			transect_fileID = selected_pd_item.iloc[0]['Transect_fileID']
			user_ID = selected_pd_item.iloc[0]['User_ID']
			rating = selected_pd_item.iloc[0]['Rating']
			image_type = selected_pd_item.iloc[0]['Image_Type']
			image_format = selected_pd_item.iloc[0]['Image_Format']

		if image_type == "3D":
			transect_path = os.path.join(self.img_path, "projections", transect_fileID)
			try:
				if "KAH_SIO" in transect_fileID:
					transect_file = glob.glob(transect_path + "/*KAH_SIO*")[0]
				elif "Siderastreidae" in transect_fileID:
					transect_file = glob.glob(transect_path + "/*Siderastreidae*")[0]
				elif "Guam_RedAlgae" in transect_fileID:
					transect_file = glob.glob(transect_path + "/*RedAlgae*")[0]
				else:
					transect_file = glob.glob(transect_path + "/*transect*")[0]
			except:
				print("Transect File for " + transect_fileID + " not found!")
				raise FileNotFoundError
			transect_path = transect_file
		else:
			temp_transect_path = selected_pd_item.iloc[0]['Transect_Path']
			transect_path = os.path.join(self.img_path, "coral-transect", *temp_transect_path, transect_fileID + "." + image_format)

		img = cv2.imread(transect_path, cv2.IMREAD_COLOR)
		if img is None:
			print("File not found at " + transect_path + "!")
			raise FileNotFoundError
		elif image_type == "3D":
			img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
		elif image_type == "satellite":
			if self.satellite_df is None:
				print("Satellite dataframe not initialized!")
				raise ValueError
			selected_sat_item = self.satellite_df[self.satellite_df["Transect_fileID"] == transect_fileID + "." + image_format]
			mosaicpath = os.path.join(self.img_path, "Fiji_Transects", selected_sat_item.iloc[0]["Mosaic_ID"])
			ystart = selected_sat_item.iloc[0]["ystart"]
			xstart = selected_sat_item.iloc[0]["xstart"]

			if os.path.exists(mosaicpath):
				mosaicdata = coralutils.CoralData(mosaicpath, load_type="raster")
			else:
				print("Transect File for " + transect_fileID + " not found!")
				raise FileNotFoundError
			img = np.copy(mosaicdata.image[ystart+128:ystart+384, xstart+128:xstart+384, :]) # This is all 8 channel data (BGR+NIR order)

		return img

	def load_classification(self, classification_fileID: str, transect_fileID: str = None, filter_area: int = 5) -> CoralImage:
	# load given image, and filter it by given filter area

		# Try to find classification image record in loaded pd dataframe
		selected_pd_item = self.NeMO_df[self.NeMO_df['Classification_fileID'] == classification_fileID]
		if len(selected_pd_item) == 0:
			print(classification_fileID + " not found in loaded pandas dataframe... cannot load file!")
			raise FileNotFoundError
		else:
			transect_fileID = selected_pd_item.iloc[0]['Transect_fileID']
			user_ID = selected_pd_item.iloc[0]['User_ID']
			rating = selected_pd_item.iloc[0]['Rating']
			image_type = selected_pd_item.iloc[0]['Image_Type']

		# load RGB classification image
		if image_type is "3D":
			transect_path = os.path.join(self.img_path, "projections", transect_fileID)
			filepath = os.path.join(transect_path, "proj_" + classification_fileID + ".png")
		else:
			transect_path = os.path.join(self.img_path, "coral-transect-classifications")
			filepath = os.path.join(transect_path, classification_fileID)
		img = self.load_transect_file(classification_fileID)

		img_class_RGB = cv2.imread(filepath, cv2.IMREAD_COLOR)
		if img_class_RGB is None:
			print("Classification file not found at " + filepath + "!")
			raise FileNotFoundError
		img_class_RGB = cv2.cvtColor(img_class_RGB, cv2.COLOR_BGR2RGB)
		ysize, xsize, channels = img_class_RGB.shape

		# find all areas that have something classified, or is out of bounds
		if image_type == "3D":
			img_filter = np.zeros((ysize, xsize)) # coral_dict classes
			for k in self.coralcolors3D_dict:
				c = self.coralcolors3D_dict[k].lstrip("#")
				c_RGB = tuple(int(c[i:i+2], 16) for i in (0, 2, 4)) # convert to RGB tuple
				# if k == "OutofBounds":
				# 	y,x = np.where(np.all(img_class_RGB == np.asarray(c_RGB, dtype=np.uint8), axis=-1))
				# 	img_class_RGB[y,x] = np.asarray(c_RGB, dtype=np.uint8)
				if k == "Acroporidae": # older classification color scheme
					y,x = np.where(np.all(img_class_RGB == self.alternateColors_dict[k], axis=-1))
					img_class_RGB[y,x] = np.asarray(c_RGB, dtype=np.uint8) # change to standardized color scheme

				y,x = np.where(np.all(img_class_RGB == np.asarray(c_RGB, dtype=np.uint8), axis=-1)) # find classes that have color c_RGB[k]
				img_filter[y,x] = 1 # Mark all classes that are in coral_dict as 1 (this will filter out boundaries between classes as well)

			# fill in unclassified areas with surroudingarea KNN
			surroundingarea = filter_area      
			yy, xx = np.where(img_filter == 0)

			for j,i in zip(yy,xx):
				crop_len = int((surroundingarea-1)/2)
				found_replace = False
				while found_replace is False:
					tempy_min = max(j-crop_len, 0)
					tempy_max = min(j+crop_len+1, ysize)
					tempx_min = max(i-crop_len, 0)
					tempx_max = min(i+crop_len+1, xsize)
					truthmap_patch = img_class_RGB[tempy_min:tempy_max, tempx_min:tempx_max,:]
					unq, unq_count = np.unique(truthmap_patch.reshape(-1, truthmap_patch.shape[2]), return_counts=True, axis=0)

					if len(unq) > 0:
						found_replace = True
					else:
						crop_len += 1
				maxidx = np.argmax(unq_count)
				img_class_RGB[j,i] = unq[maxidx]

			# Find areas that still have no data or unknown
			img_perc_filter = np.ones((ysize, xsize)) # all classes except No Data, OutofBounds or Unknown
			for k in self.coralcolors3D_dict:
				c = self.coralcolors3D_dict[k].lstrip("#")
				c_RGB = tuple(int(c[i:i+2], 16) for i in (0, 2, 4)) # convert to RGB tuple
				if k == "No Data" or k == "Unknown":
				    y,x = np.where(np.all(img_class_RGB == np.asarray(c_RGB, dtype=np.uint8), axis=-1))
				    img_perc_filter[y,x] = 0

			# Transform to 0 to k classes (categorical)
			img_class = np.zeros((ysize, xsize))
			for k in self.coralcolors3D_dict.keys():
				c = self.coralcolors3D_dict[k].lstrip("#")
				c_RGB = tuple(int(c[i:i+2], 16) for i in (0, 2, 4)) # convert to RGB tuple
				y,x = np.where(np.all(img_class_RGB == np.asarray(c_RGB, dtype=np.uint8), axis=-1))
				img_class[y,x] = self.coral3D_dict[k]

			num_classes = len(self.coralcolors3D_dict)
		elif image_type == "satellite":
			white = np.asarray([0,0,0])
			surroundingarea = filter_area
			if surroundingarea % 2 == 0:
				raise ValueError('Please choose an odd number for fill_in_truthmap surroundingarea')
			y,x = np.where(np.all(img_class_RGB == white, axis=-1))
			for j,i in zip(y,x):
				crop_len = int((surroundingarea-1)/2)
				found_replace = False
				while found_replace is False:
					tempy_min = max(j-crop_len,0)
					tempy_max = min(j+crop_len+1, img_class_RGB.shape[0])
					tempx_min = max(i-crop_len,0)
					tempx_max = min(i+crop_len+1, img_class_RGB.shape[1])
					truthmap_patch = img_class_RGB[tempy_min:tempy_max,tempx_min:tempx_max,:]
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

				# Transform to 0 to k classes (categorical)
			img_class = np.zeros((ysize, xsize))
			for k in self.satellitecolors_dict.keys():
				c = self.satellitecolors_dict[k].lstrip("#")
				c_RGB = tuple(int(c[i:i+2], 16) for i in (0, 2, 4)) # convert to RGB tuple
				y,x = np.where(np.all(img_class_RGB == np.asarray(c_RGB, dtype=np.uint8), axis=-1))
				img_class[y,x] = self.coralsat_dict[k]

			img_perc_filter = ysize*xsize # 100% filled
			num_classes = len(self.satellitecolors_dict)

		return CoralImage(
			image = img,
			classification = img_class,
			user_ID = user_ID,
			transect_name = transect_fileID,
			classification_ID = classification_fileID,
			rating = rating,
			perc_classified = np.sum(img_perc_filter)/(ysize*xsize),
			num_classes = num_classes)

	def filter_by_transect(self, input_pd = None, transect_num = 0):
		# Filter transects by # of classifications available per transect
		if input_pd is None:
			filtered_pd = self.NeMO_df.pivot_table(index=['Transect_fileID'], aggfunc='size')
		else:
			filtered_pd = input_pd.pivot_table(index=['Transect_fileID'], aggfunc='size')
		pd_idx = np.where(filtered_pd >= transect_num)[0]
		filtered_pd_vals = filtered_pd.index[pd_idx]
		filtered_pd = self.NeMO_df.loc[self.NeMO_df["Transect_fileID"].isin(filtered_pd_vals)]

		return filtered_pd, filtered_pd_vals

	def filter_by_rating(self, input_pd = None, rating_range = [0,0] , include_nan = False):
		# Filter transects by rating value with rating_range (inclusive)
		if input_pd is None:
			input_pd = self.NeMO_df

		rating_idx = np.ones(len(input_pd))
		if rating_range[0] == rating_range[1]:
			rating_idx = rating_idx & (input_pd["Rating"] == rating_range[0])
		elif rating_range[0] < rating_range[1]:
			rating_idx = rating_idx & (input_pd["Rating"] >= rating_range[0]) & (input_pd["Rating"] <= rating_range[1])
		else:
			print("Error! Check rating range!")
			raise ValueError

		if include_nan:
			rating_idx = rating_idx & np.isnan(input_pd["Rating"])

		return input_pd[rating_idx]

	def filter_by_user(self, input_pd = None, transect_num = 0, user_rating=None):
		# Filter transects by # of classifications performed by each user
		# Optional: Specify minimum average user_rating that each user must hit

		if input_pd is None:
			filtered_pd = self.NeMO_df.pivot_table(index=['User_ID'], aggfunc='size')
		else:
			filtered_pd = input_pd.pivot_table(index=['User_ID'], aggfunc='size')

		pd_idx = np.where(filtered_pd >= transect_num)[0]
		if user_rating is not None:
			rating_pd_idx = []
			for idx in pd_idx:
				sub_userdf = self.NeMO_df.loc[self.NeMO_df['User_ID'] == filtered_pd.index[idx]]
				if np.mean(sub_userdf['Rating']) >= user_rating:
					rating_pd_idx.append(idx)
			pd_idx = rating_pd_idx

		filtered_pd_vals = filtered_pd.index[pd_idx]
		filtered_pd = self.NeMO_df.loc[self.NeMO_df["User_ID"].isin(filtered_pd_vals)]

		return filtered_pd, filtered_pd_vals

	def median_stack_img(self, input_pd):
		if len(input_pd.Transect_fileID.unique()) != 1:
			print("Error! Not a unique transect dataframe!")
			raise ValueError

		img_class_list = []
		for index, row in input_pd.iterrows():
			try:
				img_class = self.load_classification(row["Classification_fileID"], row["Transect_fileID"])
				img_class_list.append(img_class)
			except FileNotFoundError:
				continue

		img_class_stack = np.stack(img_class_list, axis=2)
		img_class_median = np.median(img_class_stack, axis=-1)
		img_median_dev = np.abs(img_class_stack - img_class_median[:,:,np.newaxis])
		img_median_dev[img_median_dev > 0] = 1
		img_median_dev = np.mean(img_median_dev, axis=-1)

		return img_class_median, img_median_dev

if __name__ == '__main__':
	classify_masterfile = "/home/rechant/Documents/NeMO-Net/NeMO-NET/Images/AppData_May/NeMO-MayCSVs/ClassificationAttempts_May_Master.csv"
	img_path = "/home/rechant/Documents/NeMO-Net/NeMO-NET/Images/AppData_May"
	NeMO_Data = Coral3DData(classify_masterfile, img_path = img_path, meshType = ["3D"])
	NeMO_Data.load_satellite_transect_data()

	user_pd, user_pd_vals = NeMO_Data.filter_by_user(transect_num=20, user_rating=0.5) # only find users with more than 20 transect classifications

	# usr_id = 'fb67f1a3-9388-11e9-892b-0239699f2190' # Alan
	# # usr_id = '054e4b47-92ba-11e9-892b-0239699f2190' # Juan
	# sub_userdf = NeMO_Data.NeMO_df.loc[NeMO_Data.NeMO_df['User_ID'] == usr_id]

	# CIL = CoralImageList()

	# for i in range(len(sub_userdf)):
	# 	classification_fileID = sub_userdf.iloc[i]['Classification_fileID']
	# 	transect_fileID = sub_userdf.iloc[i]['Transect_fileID']

	# 	try:
	# 		coralimage = NeMO_Data.load_classification(classification_fileID = classification_fileID)
	# 		CIL.append(coralimage)
	# 		print(classification_fileID + " loaded, corresponding to " + transect_fileID)
	# 	except FileNotFoundError:
	# 		print("Skipping " + classification_fileID + ", corresponding to " + transect_fileID + "!")

	# CIL.output_training(directory = "/home/rechant/Documents/NeMO-Net/NeMO-NET/Images/AppData_Training", add_to_dir = False)
