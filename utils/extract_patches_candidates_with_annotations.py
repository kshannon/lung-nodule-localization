#! /usr/bin/env python

# example patch call:
# ./extract_patches.py -subset 202 -slices 64 -dim 64

#### ---- Imports & Dependencies ---- ####
import sys
import os
import argparse
from configparser import ConfigParser
import pathlib
from glob import glob
from random import shuffle
import SimpleITK as sitk # pip install SimpleITK
from tqdm import tqdm # pip install tqdm
import h5py
import pandas as pd
import numpy as np
from scipy.spatial import distance


#### ---- Argparse Utility ---- ####
parser = argparse.ArgumentParser(description='Modify the patch extractor script',add_help=True)
parser.add_argument('-hdf5',
					action="store_true",
					dest="hdf5",
					default=True,
					help='Save processed data to hdf5')
parser.add_argument('-hu_norm',
					action="store_true",
					dest="hu_norm",
					default=False,
					help='Normalize Patch to -1000 - 400 HU')
parser.add_argument('-slices',
					type=int,
					action="store",
					dest="slices",
					default=1,
					help='Num of tensor slices > 0, default = 1')
parser.add_argument('-dim',
					action="store",
					dest="dim",
					type=int,
					default=64,
					help='Dimension of the patch, default = 64')
parser.add_argument('-remote',
					action="store_true",
					dest="remote",
					default=False,
					help='Use if running script remote e.g. AWS')

requiredNamed = parser.add_argument_group('required named arguments')
requiredNamed.add_argument('-subset',
							action="store",
							dest="subset",
							type=lambda s: ['subset'+str(x)+'/' for x in s.split(',')],
							required=True,
							help='subset dir name or number(s) e.g. 0,1,2')
args = parser.parse_args()


#### ---- ConfigParse Utility ---- ####
config = ConfigParser()
config.read('extract_patches_config.ini') #local just for now (need if - else for AWS)
'''
Example extract_patches_config.ini file:
	[local]
	LUNA_PATH = /Users/keil/datasets/LUNA16/
	CSV_PATH = /Users/keil/datasets/LUNA16/csv-files/
	IMG_PATH = /Users/keil/datasets/LUNA16/patches/
	[remote]
	# - when we move to AWS
'''

#### ---- Global Vars ---- ####
LUNA_PATH = config.get('local', 'LUNA_PATH')
CSV_PATH = config.get('local', 'CSV_PATH')
IMG_PATH = config.get('local', 'IMG_PATH')
SUBSET = args.subset
SAVE_HDF5 = args.hdf5
HU_NORM = args.hu_norm
PATCH_DIM = args.dim
NUM_SLICES = args.slices
CHANNELS = 1
PATCH_WIDTH = PATCH_DIM/2
PATCH_HEIGHT = PATCH_DIM/2
PATCH_DEPTH = NUM_SLICES/2
# WORK_REMOTE = args.remote #add later w/ AWS
#TODO add this to config file for csv file name
DF_NODE = pd.read_csv(CSV_PATH + "candidates_V2.csv")
DF_NODE1 = pd.read_csv(CSV_PATH + "candidates_with_annotations.csv")

# DF_NODE = pd.read_csv(CSV_PATH + "candidates_with_annotations.csv")
FILE_LIST = []
SUBSET_LIST = []
for unique_set in SUBSET:
	mhd_files = glob("{}{}/*.mhd".format(LUNA_PATH, unique_set))
	FILE_LIST.extend(mhd_files) #add subset of .mhd files
	subset_num = unique_set.strip('subset/') #extracting out subset number
	for elements in mhd_files: #making sure we match each globbed mhd file to a subset num
		SUBSET_LIST.append(int(subset_num)) #pass this list later to write subset num to HDF5


#### ---- Helper Functions ---- ####
def normalizePlanes(npzarray):
	"""
	Normalize pixel depth into Hounsfield units (HU), between -1000 - 400 HU
	All other HU will be masked. Then we normalize pixel values between 0 and 1.
	"""
	maxHU, minHU = 400., -1000.
	npzarray = (npzarray - minHU) / (maxHU - minHU)
	npzarray[npzarray>1] = 1.
	npzarray[npzarray<0] = 0.
	return npzarray

def normalize_img(img):
	"""
	Sets the MHD image to be approximately 1.0 mm voxel size
	https://itk.org/ITKExamples/src/Filtering/ImageGrid/ResampleAnImage/Documentation.html
	"""
	# Number of pixels you want for x,y,z dimensions
	new_x_size = int(img.GetSpacing()[0]*img.GetWidth())
	new_y_size = int(img.GetSpacing()[1]*img.GetHeight())
	new_z_size = int(img.GetSpacing()[2]*img.GetDepth())
	new_size = [new_x_size, new_y_size, new_z_size]
	new_spacing = [1,1,1]  # New spacing to be 1.0 x 1.0 x 1.0 mm voxel size

	interpolator_type = sitk.sitkBSpline #sitkLinear using BSpline over Linear
	return sitk.Resample(img, np.array(new_size, dtype='uint32').tolist(),
							sitk.Transform(),
							interpolator_type,
							img.GetOrigin(),
							new_spacing,
							img.GetDirection(),
							0.0,
							img.GetPixelIDValue())

def make_bbox(center,width,height,depth,origin,class_id):
	"""
	Returns a 3d (numpy tensor) bounding box from the CT scan.
	2d in the case where PATCH_DEPTH = 1
	"""
	# left = np.max([0, np.abs(center[0] - origin[0]) - PATCH_WIDTH]).astype(int)
	left = np.max([0, center[0] - PATCH_WIDTH]).astype(int)
	right = np.min([width, center[0] + PATCH_WIDTH]).astype(int)
	# left = int((np.abs(center[0] - origin[0])) - PATCH_WIDTH) #DEBUG
	# right = int((np.abs(center[0] - origin[0])) + PATCH_WIDTH) #DEBUG
	down = np.max([0, center[1] - PATCH_HEIGHT]).astype(int)
	up = np.min([height, center[1] + PATCH_HEIGHT]).astype(int)
	top = np.min([depth, center[2] + PATCH_DEPTH]).astype(int)
	bottom = np.max([0, center[2] - PATCH_DEPTH]).astype(int)

	bbox = [[down, up], [left, right], [bottom, top]] #(back,abdomen - left side, right side - feet, head)

	# If bbox has a origin - center - PATCH_DIM/2 that results in a 0, (rarely the case)
	# ensure that the bbox dims are all [PATCH_DIM x PATCH_DIM x PATCH_DIM]
	if bbox[0][0] == 0:
		bbox[0][1] = PATCH_DIM
	elif bbox[1][0] == 0:
		bbox[1][1] = PATCH_DIM
	elif bbox[2][0] == 0:
		bbox[2][1] = NUM_SLICES # change to --slice dim

	return bbox

def downsample_class_0(df):
	"""
	Returns a pd.DataFrame where class 0s that collide with class 1s
	have been flagged based on a distance measurement threshold.
	Threshold = PATCH_DIM/2
	The flag will be written to HDF5 and let the user know not to train on these class 0s
	"""
	empty_col =  [0 for x in range(len(df))]
	idx_to_flag = []
	df.reset_index(inplace=True)
	if 1 in df['class'].tolist(): #check series ID for a positive nodule
		df_class_1 = df[df["class"] == 1].copy(deep=True)
		ones_coords = df_class_1[["coordX", "coordY", "coordZ"]].values
		for idx, row in df.iterrows():
			#check for a class 1
			if row['class'] == 1:
				continue
			#set vars for calculation
			zero_coord = (row['coordX'],row['coordY'],row['coordZ'])
			for one_coord in ones_coords:
				dst = distance.euclidean(zero_coord,one_coord)
				if dst <= PATCH_DIM/2: #follow this heuristic for downsampling class 0
					idx_to_flag.append(idx)

	else:
		df = df.assign(no_train = empty_col)
		return df


	idx_to_flag = list(set(idx_to_flag))
	downsample_col = []
	for idx, i in enumerate(empty_col):
		if idx in idx_to_flag:
			downsample_col.append(1)
		else:
			downsample_col.append(0)

	df = df.assign(no_train = downsample_col)
	return df

def write_to_hdf5(dset_and_data,first_patch=False):
	"""Accept zipped hdf5 dataset obj and numpy data, write data to dataset"""
	dset = dset_and_data[0] #hdf5 dataset obj
	data = dset_and_data[1] #1D numpy hdf5 writable data

	if first_patch == True:
		dset[:] = data #set the whole, empty, hdf5 dset = data
		return
	row = dset.shape[0] # Count current dataset rows
	dset.resize(row+1, axis=0) # Add new row
	dset[row, :] = data # Insert data into new row
	return


#### ---- Process CT Scans and extract Patches (the pipeline) ---- ####
def main():
	"""
	Create the hdf5 file + datasets, iterate thriough the folders DICOM imgs
	Normalize the imgs, create mini patches and write them to the hdf5 file system
	"""
	with h5py.File(LUNA_PATH + str(PATCH_DIM) + 'x' + str(PATCH_DIM) + 'x' + str(NUM_SLICES) + '-patch.hdf5', 'w') as HDF5:
		# Datasets for 3d patch tensors & class_id/x,y,z coords
		total_patch_dim = PATCH_DIM * PATCH_DIM * NUM_SLICES
		patch_dset = HDF5.create_dataset('input', (1,total_patch_dim), maxshape=(None,total_patch_dim)) #patches = inputs
		class_dset = HDF5.create_dataset('output', (1,1), maxshape=(None,1), dtype=int) #classes = outputs
		notrain_dset = HDF5.create_dataset('notrain', (1,1), maxshape=(None,1), dtype=int) # test holdout
		centroid_dset = HDF5.create_dataset('centroid', (1,3), maxshape=(None,3), dtype=float)
		uuid_dset = HDF5.create_dataset('uuid', (1,1), maxshape=(None,None), dtype=h5py.special_dtype(vlen=bytes))
		subset_dset = HDF5.create_dataset('subsets', (1,1), maxshape=(None,1), dtype=int)
		diameter_dset = HDF5.create_dataset('diameter_label', (1,1), maxshape=(None,1), dtype=int)
		HDF5['input'].attrs['lshape'] = (PATCH_DIM, PATCH_DIM, NUM_SLICES, CHANNELS) # (Height, Width, Depth)
		print("Successfully initiated the HDF5 file. Ready to recieve data!")

		#### ---- Iterating through a CT scan ---- ####
		counter = 0
		scan_number = 1
		first_patch = True # flag for saving first img to hdf5
		for img_file, subset_id in tqdm(zip(FILE_LIST,SUBSET_LIST)):
			print("Processing CT Scan: {}".format(scan_number))
			base=os.path.basename(img_file)  # Strip the filename out
			seriesuid = os.path.splitext(base)[0]  # Get the filename without the extension
			mini_df = DF_NODE[DF_NODE["seriesuid"] == seriesuid]

			#### ---- Downsampling Class 0s ---- ####
			mini_df = downsample_class_0(mini_df)

			mini_df1 = DF_NODE1[DF_NODE1["seriesuid"] == seriesuid]
			mini_df1.fillna(-1, inplace=True)

			# Load the CT scan (3D .mhd file)
			# Numpy is z,y,x and SimpleITK is x,y,z -- (note the ordering of dimesions)
			itk_img = sitk.ReadImage(img_file)

			# Normalize the image spacing so that a voxel is 1x1x1 mm in dimension
			itk_img = normalize_img(itk_img)

			# SimpleITK keeps the origin and spacing information for the 3D image volume
			img_array = sitk.GetArrayFromImage(itk_img) # indices are z,y,x (note the ordering of dimesions)
			img_array = np.pad(img_array, int(PATCH_DIM), mode="constant", constant_values=-2000)#, constant_values=0) #0 padding 3d array for patch clipping issue
			slice_z, height, width = img_array.shape
			origin = np.array(itk_img.GetOrigin())      # x,y,z  Origin in world coordinates (mm) - Not same as img_array
			spacing = np.array(itk_img.GetSpacing())    # spacing of voxels in world coordinates (mm)
			scan_number += 1


			#### ---- Iterating through a CT scan's slices ---- ####
			for candidate_idx, cur_row in mini_df.iterrows(): # Iterate through all candidates (in dataframe)
				# This is the real world x,y,z coordinates of possible nodule (in mm)
				class_id = cur_row["class"] #0 for false, 1 for true nodule
				no_train = cur_row["no_train"]
				candidate_x = cur_row["coordX"] + PATCH_DIM
				candidate_y = cur_row["coordY"] + PATCH_DIM
				candidate_z = cur_row["coordZ"] + PATCH_DIM
				center = np.array([candidate_x, candidate_y, candidate_z])   # candidate center
				voxel_center = np.rint(np.abs(center / spacing - origin)).astype(int)  # candidate center in voxels

				if class_id == 0:
					diameter_label = 0
				else:
					diameter_label = mini_df1[(mini_df1['coordX']==cur_row["coordX"]) & (mini_df1['coordY']==cur_row["coordY"]) & (mini_df1['coordZ']==cur_row["coordZ"])]['diameter_mm'].values
					if len(diameter_label) == 0:
						diameter_label = -1
						no_train = 1
						# put a counter here

				#### ---- Generating the 2d/2.5d/3d Patch ---- ####
				bbox = make_bbox(voxel_center, width, height, slice_z, origin, class_id) #return bounding box
				patch = img_array[
					bbox[2][0]:bbox[2][1],
					bbox[0][0]:bbox[0][1],
					bbox[1][0]:bbox[1][1]]

				# DEBUG print(patch.shape) #uncomment to debug shape size being written

				#### ---- Perform Hounsfield Normlization ---- ####
				if HU_NORM:
					patch = normalizePlanes(patch) #normalize patch to HU units


				#### ---- Prepare Data for HDF5 insert ---- ####
				patch = patch.ravel().reshape(1,-1) #flatten img to (1 x N)
				if patch.shape[1] != total_patch_dim: # Catch any class 0 bbox issues and pass them
					counter += 1
					continue
				#minor fix to subtract the PATCH_DIM from each centroid when saving to HDF5 to match candidates_V2.csv
				centroid_data = np.array([candidate_x - PATCH_DIM,candidate_y - PATCH_DIM,candidate_z - PATCH_DIM]).ravel().reshape(1,-1)
				seriesuid_str = np.string_(seriesuid) #set seriesuid str to numpy.bytes_ type





				#### ---- Write Data to HDF5 insert ---- ####
				hdf5_dsets = [patch_dset, class_dset, notrain_dset, uuid_dset, subset_dset, centroid_dset,diameter_dset]
				hdf5_data = [patch, class_id, no_train, seriesuid_str, subset_id, centroid_data,diameter_label]

				for dset_and_data in zip(hdf5_dsets,hdf5_data):
					if first_patch == True:
						write_to_hdf5(dset_and_data,first_patch=True)
					else:
						write_to_hdf5(dset_and_data)
				first_patch = False

	print("Did not write: " + str(counter) + " patches to HDF5")
	print("All {} CT Scans Processed and Individual Patches written to HDF5!".format(scan_number))
	print('\a')

if __name__ == '__main__':
	main()
