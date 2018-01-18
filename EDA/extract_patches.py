#! /usr/bin/env python

# Thanks to Jonathan Mulholland and Aaron Sander from Booz Allen Hamilton who
# made their code publically availble, parts of which we are using in this script.
# https://www.kaggle.com/c/data-science-bowl-2017/details/tutorial

#TODO: determine if voxel edge detection is a suffucient issue to solve.


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
from scipy.misc import imsave # conda install Pillow or PIL


#### ---- Argparse Utility ---- ####
parser = argparse.ArgumentParser(description='Modify the patch extractor script',add_help=True)
parser.add_argument('-img',
					action="store_true",
					dest="img",
					default=False,
					help='Save .png patches to ./patches/')
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
					help='Dimmension of the patch, default = 64')
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

# Example extract_patches_config.ini file:
	# [local]
	# LUNA_PATH = /Users/keil/datasets/LUNA16/
	# CSV_PATH = /Users/keil/datasets/LUNA16/csv-files/
	# IMG_PATH = /Users/keil/datasets/LUNA16/patches/
	# [remote]
	# # - when we move to AWS


#### ---- Global Vars ---- ####
LUNA_PATH = config.get('local', 'LUNA_PATH')
CSV_PATH = config.get('local', 'CSV_PATH')
IMG_PATH = config.get('local', 'IMG_PATH')
SUBSET = args.subset
SAVE_IMG = args.img
HU_NORM = args.hu_norm
PATCH_DIM = args.dim
NUM_SLICES = args.slices
# This is really the half (width,height,depth) so window will be double these values
PATCH_WIDTH = PATCH_DIM/2
PATCH_HEIGHT = PATCH_DIM/2
PATCH_DEPTH = NUM_SLICES/2
# WORK_REMOTE = args.remote #add later w/ AWS
DF_NODE = pd.read_csv(CSV_PATH + "candidates_with_annotations.csv")
FILE_LIST = []
for unique_set in SUBSET:
	FILE_LIST.extend(glob("{}{}/*.mhd".format(LUNA_PATH, unique_set))) #add subset of .mhd files


#### ---- Helper Functions ---- ####
def normalizePlanes(npzarray):
	"""
	Normalize pixel depth into Hounsfield units (HU), between -1000 - 400 HU
	All other HU will be masked. Then we normalize pixel values between 0 and 1.
	"""
	maxHU, minHU = 400., 1000.
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


def make_bbox(center,width,height,depth,origin):
	"""
	Returns a 3d (numpy tensor) bounding box from the CT scan.
	2d in the case where PATCH_DEPTH = 1
	"""
	# TODO:  The height and width seemed to be switched. Simplify if possible
	left = np.max([0, np.abs(center[0] - origin[0]) - PATCH_WIDTH]).astype(int)
	right = np.min([width, np.abs(center[0] - origin[0]) + PATCH_WIDTH]).astype(int)
	# left = int((np.abs(center[0] - origin[0])) - PATCH_WIDTH) #DEBUG
	# right = int((np.abs(center[0] - origin[0])) + PATCH_WIDTH) #DEBUG
	down = np.max([0, np.abs(center[1] - origin[1]) - PATCH_HEIGHT]).astype(int)
	up = np.min([height, np.abs(center[1] - origin[1]) + PATCH_HEIGHT]).astype(int)
	top = np.min([depth, np.abs(center[2] - origin[2]) + PATCH_DEPTH]).astype(int)
	bottom = np.max([0, np.abs(center[2] - origin[2]) - PATCH_DEPTH]).astype(int)

	bbox = [[down, up], [left, right], [bottom, top]] #(back,abdomen - left side, right side - feet, head)
	return bbox

def save_img():
	#TODO
	pass

def write_to_hdf5():
	#TODO
	pass


#### ---- Process CT Scans and extract Patches (the pipeline) ---- ####
def main():
	"""
	Create the hdf5 file + datasets, iterate thriough the folders DICOM imgs
	Normalize the imgs, create mini patches and write them to the hdf5 file system
	"""
	with h5py.File(LUNA_PATH + str(PATCH_DIM) + 'dim_patches.hdf5', 'w') as HDF5:
		# Datasets for 3d patch tensors & class_id/x,y,z coords
		total_patch_dim = PATCH_DIM * PATCH_DIM * NUM_SLICES
		img_dset = HDF5.create_dataset('patches', (1,total_patch_dim), maxshape=(None,total_patch_dim))
		class_dset = HDF5.create_dataset('classes', (1,4), maxshape=(None,4), dtype=float)
		uuid_dset = HDF5.create_dataset('uuid', (1,1), maxshape=(None,None), dtype=h5py.special_dtype(vlen=bytes)) #old one
		print("Created HDF5 File and Three Datasets")

		#### ---- Iterating through a CT scan ---- ####
		first_patch = True # flag for saving first img to hdf5
		for img_file in tqdm(FILE_LIST):

			base=os.path.basename(img_file)  # Strip the filename out
			seriesuid = os.path.splitext(base)[0]  # Get the filename without the extension
			mini_df = DF_NODE[DF_NODE["seriesuid"] == seriesuid]

			# Load the CT scan (3D .mhd file)
			# Numpy is z,y,x and SimpleITK is x,y,z -- (note the ordering of dimesions)
			itk_img = sitk.ReadImage(img_file)

			# Normalize the image spacing so that a voxel is 1x1x1 mm in dimension
			itk_img = normalize_img(itk_img)

			# SimpleITK keeps the origin and spacing information for the 3D image volume
			img_array = sitk.GetArrayFromImage(itk_img) # indices are z,y,x (note the ordering of dimesions)
			img_array = np.pad(img_array, int(PATCH_DIM/2), mode="constant", constant_values=0) #0 padding 3d array for patch clipping issue
			slice_z, height, width = img_array.shape
			origin = np.array(itk_img.GetOrigin())      # x,y,z  Origin in world coordinates (mm) - Not same as img_array
			spacing = np.array(itk_img.GetSpacing())    # spacing of voxels in world coordinates (mm)


			#### ---- Iterating through a CT scan's slices ---- ####
			for candidate_idx, cur_row in mini_df.iterrows(): # Iterate through all candidates (in dataframe)
				# This is the real world x,y,z coordinates of possible nodule (in mm)
				class_id = cur_row["class"] #0 for false, 1 for true nodule
				diam = cur_row["diameter_mm"]  # Only defined for true positives
				if np.isnan(diam):
					diam = 30.0  # If NaN, then just use a default of 30 mm

				candidate_x = cur_row["coordX"]
				candidate_y = cur_row["coordY"]
				candidate_z = cur_row["coordZ"]
				center = np.array([candidate_x, candidate_y, candidate_z])   # candidate center
				#TODO ask tony/research why we are subtracting ct scan origin from ROI centert, looks like stnd norm


				#### ---- Generating the Patch ---- ####
				bbox = make_bbox(center, width, height, slice_z, origin) #return bounding box
				patch = img_array[
					bbox[0][0]:bbox[0][1],
					bbox[1][0]:bbox[1][1],
					bbox[2][0]:bbox[2][1]]


				#### ---- Writing patch.png to patches/ ---- ####
				#TODO 3d --> 2d and save img
				if SAVE_IMG: # only if -img flag is passed
					imsave(IMG_PATH + "class_{}_uid_{}_xyz_{}_{}_{}.png".format(
							class_id,
							seriesuid,
							candidate_x,
							candidate_y,
							candidate_z), patch)


				#### ---- Prepare Data for HDF5 insert ---- ####
				if HU_NORM:
					patch = normalizePlanes(patch) #normalize patch to HU units
				patch = patch.ravel().reshape(1,-1) #flatten img to (1 x N)
				# Flatten class, and x,y,z coords into vector for storage
				meta_data = np.array([float(class_id),candidate_x,candidate_y,candidate_z]).ravel().reshape(1,-1)
				seriesuid_str = np.string_(seriesuid) #set seriesuid str to numpy.bytes_ type

				## -- DEBUG -- ##
				# if statement to catch patch clipping issues, uncomment to debug
				# if patch.shape[1] == 0:
					# print("Patch Clipped: {}".format(patch.shape))
				# 	continue

				## -- DEBUG --##
				# Y-axis issue with bbox, possibly due to patient axial position during CT Scan
				# More info is needed to resolve this small Data integrity issue
				# Note this issue DOES NOT effect any class 1 Pathes. Therefore we skip these for now.
				# Recommend to confirm this hypothesis for scan w/ many class 1s
				# Suggest this scan:
				if patch.shape[1] != total_patch_dim and patch.shape[1] != 0:
					print("--- Bad Actor Found! ---")
					print("class ID: " + str(class_id))
					print("origin: " + str(origin))
					print("center: " + str(center))
					print("img array shape" + str(img_array.shape))
					print("patch shape: " + str(patch.shape[1]))
					print("bbox: " + str(bbox))
					print("------------------------")
					continue


				#### ---- Write Data to HDF5 insert ---- ####
				if first_patch == True: # For first patch only
					img_dset[:] = patch
					class_dset[:] = meta_data
					uuid_dset[:] = seriesuid_str
					first_patch = False
					print("Images Being Saved to HDF5!")

				else:
					if center[0] < -98.9: #and center[1] < -89.9:
						print("center: " + str(center))
					row = img_dset.shape[0] # Count current dataset rows
					img_dset.resize(row+1, axis=0) # Add new row
					img_dset[row, :] = patch # Insert data into new row

					row = class_dset.shape[0]
					class_dset.resize(row+1, axis=0)
					class_dset[row, :] = meta_data

					row = uuid_dset.shape[0]
					uuid_dset.resize(row+1, axis=0)
					uuid_dset[row, :] = seriesuid_str


	print("All Images Processed and Patches written to HDF5. Thank you patch again!")
	print('\a')

if __name__ == '__main__':
	main()
