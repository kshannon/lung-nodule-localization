#! /usr/bin/env python

# Thanks to Jonathan Mulholland and Aaron Sander from Booz Allen Hamilton who
# made their code publically availble, parts of which we are using in this script.
# https://www.kaggle.com/c/data-science-bowl-2017/details/tutorial

#TODO: test 2d / 3d patches and read / reshape into numpy (ipython notebook)
#TODO: fix clipping for 2d/3d and test
#TODO: Clean Up main for loop code and comment sparingly


#### ---- Imports & Dependencies ---- ####
import sys
import os
import argparse
from configparser import ConfigParser
import pathlib
from glob import glob
from random import shuffle
import SimpleITK as sitk   # pip install SimpleITK
from tqdm import tqdm    # pip install tqdm
import h5py
import pandas as pd
import numpy as np
from scipy.misc import imsave # might require: conda install Pillow or PIL


#### ---- Argparse Utility ---- ####
parser = argparse.ArgumentParser(description='Modify the patch extractor script',add_help=True)
parser.add_argument('-img', action="store_true", dest="img", default=False,
						help='Save .png patches to ./patches/')
parser.add_argument('-slices', type=int, action="store", dest="slices",
						default=1, help='Num of tensor slices > 0, default = 1')
parser.add_argument('-dim', action="store", dest="dim", type=int, default=64,
						help='Dimmension of the patch, default = 64')
parser.add_argument('-remote', action="store_true", dest="remote", default=False,
						help='Use if running script remote e.g. AWS')
requiredNamed = parser.add_argument_group('required named arguments')
requiredNamed.add_argument('-subset', action="store", dest="subset",
						type=lambda s: ['subset'+str(x)+'/' for x in s.split(',')],
						required=True, help='subset dir name or number(s) e.g. 0,1,2')
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
PATCH_DIM = args.dim
SUBSET = args.subset
SAVE_IMG = args.img
NUM_SLICES = args.slices
# WORK_REMOTE = args.remote #add later w/ AWS
MASK_DIMS = tuple([int(PATCH_DIM/2)])*3 #set the width, height, depth, pass to make_mask()
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

	# new_spacing = [old_sz*old_spc/new_sz  for old_sz, old_spc, new_sz in zip(img.GetSize(), img.GetSpacing(), new_size)]
	new_spacing = [1,1,1]  # New spacing to be 1.0 x 1.0 x 1.0 mm voxel size
	interpolator_type = sitk.sitkLinear
	return sitk.Resample(img, np.array(new_size, dtype='uint32').tolist(),
							sitk.Transform(),
							interpolator_type,
							img.GetOrigin(),
							new_spacing,
							img.GetDirection(),
							0.0,
							img.GetPixelIDValue())

#TODO Remove make mask
def make_mask(center,diam,z,width,height,depth,spacing,origin,
			  mask_width=MASK_DIMS[0],mask_height=MASK_DIMS[1],mask_depth=MASK_DIMS[2]):

	# mask = np.zeros([height,width]) # 0"s everywhere except nodule swapping x,y to match img
	# #convert to nodule space from world coordinates
    #
	# padMask = 5 #??
    #
	# # Defining the voxel range in which the nodule falls
	# v_center = (center-origin)/spacing
	# v_diam = int(diam/spacing[0]+padMask) #TODO why diamter not radius
	# v_xmin = np.max([0,int(v_center[0]-v_diam)-padMask])
	# v_xmax = np.min([width-1,int(v_center[0]+v_diam)+padMask])
	# v_ymin = np.max([0,int(v_center[1]-v_diam)-padMask])
	# v_ymax = np.min([height-1,int(v_center[1]+v_diam)+padMask])
    #
	# v_xrange = range(v_xmin,v_xmax+1)
	# v_yrange = range(v_ymin,v_ymax+1)
    #
	# # # Convert back to world coordinates for distance calculation
	# # x_data = [x*spacing[0]+origin[0] for x in range(width)]
	# # y_data = [x*spacing[1]+origin[1] for x in range(height)]
    #
    #
	# # RECTANGULAR MASK
	# for v_x in v_xrange:
	# 	for v_y in v_yrange:
	# 		p_x = spacing[0]*v_x + origin[0]
	# 		p_y = spacing[1]*v_y + origin[1]
	# 		if ((p_x >= (center[0] - mask_width)) &
	# 			(p_x <= (center[0] + mask_width)) &
	# 			(p_y >= (center[1] - mask_height)) &
	# 			(p_y <= (center[1] + mask_height))):
    #
	# 			mask[int((np.abs(p_y-origin[1]))/spacing[1]),
	# 				int((np.abs(p_x-origin[0]))/spacing[0])] = 1.0


	# TODO:  The height and width seemed to be switched.
	# This works but needs to be simplified. It"s probably due to SimpleITK
	# versus Numpy transposed indicies.
	left = np.max([0, np.abs(center[0] - origin[0]) - mask_width]).astype(int)
	right = np.min([width, np.abs(center[0] - origin[0]) + mask_width]).astype(int)
	down = np.max([0, np.abs(center[1] - origin[1]) - mask_height]).astype(int)
	up = np.min([height, np.abs(center[1] - origin[1]) + mask_height]).astype(int)
	top = np.min([depth, np.abs(center[2] - origin[2]) + mask_depth]).astype(int)
	bottom = np.max([0, np.abs(center[2] - origin[2]) - mask_depth]).astype(int)

	bbox = [[down, up], [left, right], [bottom, top]]
	return bbox
	# return mask, bbox


def main():
	with h5py.File(LUNA_PATH + str(PATCH_DIM) + 'dim_patches.hdf5', 'w') as HDF5:
		# Datasets for 3d patch tensors & class_id/x,y,z coords
		img_dset = HDF5.create_dataset('patches', (1,PATCH_DIM*PATCH_DIM*PATCH_DIM), maxshape=(None,PATCH_DIM*PATCH_DIM*PATCH_DIM))
		class_dset = HDF5.create_dataset('classes', (1,4), maxshape=(None,4), dtype=float)
		uuid_dset = HDF5.create_dataset('uuid', (1,1), maxshape=(None,None), dtype=h5py.special_dtype(vlen=bytes)) #old one
		print("Created HDF5 File and Three Datasets")

		#### ---- Iterating through a CT scan ---- ####
		print(len(FILE_LIST))
		for img_count, img_file in enumerate(tqdm(FILE_LIST)):
			#TODO remove enumerate and set a Flag for first loop

			base=os.path.basename(img_file)  # Strip the filename out
			seriesuid = os.path.splitext(base)[0]  # Get the filename without the extension
			mini_df = DF_NODE[DF_NODE["seriesuid"] == seriesuid]


			"""
			Extracts 2D patches from the 3 planes (transverse, coronal, and sagittal).
			The sticking point here is the order of the axes. Numpy is z,y,x and SimpleITK is x,y,z.
			I've found it very difficult to keep the order correct when going back and forth,
			but this code seems to pass the sanity checks.
			"""
			# Load the CT scan (3D .mhd file)
			itk_img = sitk.ReadImage(img_file)  # indices are x,y,z (note the ordering of dimesions)

			# Normalize the image spacing so that a voxel is 1x1x1 mm in dimension
			itk_img = normalize_img(itk_img)

			# SimpleITK keeps the origin and spacing information for the 3D image volume
			img_array = sitk.GetArrayFromImage(itk_img) # indices are z,y,x (note the ordering of dimesions)


			slice_z, height, width = img_array.shape
			origin = np.array(itk_img.GetOrigin())      # x,y,z  Origin in world coordinates (mm) - Not same as img_array
			spacing = np.array(itk_img.GetSpacing())    # spacing of voxels in world coordinates (mm)


			#### ---- Iterating through a CT scan's slices ---- ####
			for candidate_idx, cur_row in mini_df.iterrows(): # Iterate through all candidates
				# This is the real world x,y,z coordinates of possible nodule (in mm)
				# Pulling out info from the DF
				candidate_x = cur_row["coordX"]
				candidate_y = cur_row["coordY"]
				candidate_z = cur_row["coordZ"]
				diam = cur_row["diameter_mm"]  # Only defined for true positives
				if np.isnan(diam):
					diam = 30.0  # If NaN, then just use a default of 30 mm
					#TODO ask tony why size = 30 mm when annotations has the max to be 32.27???
				class_id = cur_row["class"] #0 for false, 1 for true nodule

				#TODO remove mssking
				mask_width = PATCH_DIM/2 # This is really the half width so window will be double this width
				mask_height = PATCH_DIM/2 # This is really the half height so window will be double this height
				mask_depth = NUM_SLICES/2 # This is really the half depth so window will be double this depth


				center = np.array([candidate_x, candidate_y, candidate_z])   # candidate center
				#TODO ask tony/research why we are subtracting ct scan origin from ROI centert, looks like stnd norm
				voxel_center = np.rint((center-origin)/spacing).astype(int)  # candidate center in voxel space (still x,y,z ordering)

				# Calculates the bounding box (and ROI mask) for desired position
				bbox = make_mask(center, diam, voxel_center[2]*spacing[2]+origin[2],
									   width, height, slice_z, spacing, origin,
									   mask_width, mask_height, mask_depth)
				# a numpy array size of DIM x DIM
				# Confer with https://en.wikipedia.org/wiki/Anatomical_terms_of_location#Planes
				# Transverse slice 2D view - Y-X plane
				# img = img_array[bbox[2][0]:bbox[2][1],
				# 		bbox[0][0]:bbox[0][1],
				# 		bbox[1][0]:bbox[1][1]]
				# print(img.shape)

				# if TENSOR:
				# 	img_transverse = img_array[bbox[2][0]:bbox[2][1],
	            #             bbox[0][0]:bbox[0][1],
	            #             bbox[1][0]:bbox[1][1]]
					# print(img.shape) #(60,64,64) mask_depth roughly half of this value [60]
				# else:
				# img_transverse = img_array[voxel_center[2],
				# 	bbox[0][0]:bbox[0][1],
				# 	bbox[1][0]:bbox[1][1]]bbox[]



				#3d....
				img_transverse = img_array[
					bbox[0][0]:bbox[0][1],
					bbox[1][0]:bbox[1][1],
					bbox[2][0]:bbox[2][1]]
				print(type(img_transverse))
				# print(img_transverse)
				print(img_transverse.shape)
				sys.exit()



					# print(img_transverse.shape) #(64,64)

				#### ---- Writing patch.png to patches/ ---- ####
				if SAVE_IMG: # only ff -img flag is passed
					imsave(IMG_PATH + "class_{}_uid_{}_xyz_{}_{}_{}.png".format(
							class_id,
							seriesuid,
							candidate_x,
							candidate_y,
							candidate_z), img_transverse)

				# For now we will ignore imgs where the patch is getting clipped by the edge(s)
				# TODO: fix patch clipping for 3d
				img_transverse = normalizePlanes(img_transverse) #normalize HU units
				img_transverse = img_transverse.ravel().reshape(1,-1) #flatten img
				# if img_transverse.shape[1] != PATCH_DIM * PATCH_DIM:
				# 	continue


				#### ---- Writing Data to HDF5 ---- ####
				# Flatten class, and x,y,z coords into vector for storage
				meta_data = np.array([float(class_id),candidate_x,candidate_y,candidate_z]).ravel().reshape(1,-1)
				seriesuid_str = np.string_(seriesuid) #set seriesuid str to numpy.bytes_ type

				if img_count == 0: # For first patch only
					img_dset[:] = img_transverse
					class_dset[:] = meta_data
					uuid_dset[:] = seriesuid_str
				else:
					row = img_dset.shape[0] # Count current dataset rows
					img_dset.resize(row+1, axis=0) # Add new row
					img_dset[row, :] = img_transverse # Insert data into new row

					row = class_dset.shape[0]
					class_dset.resize(row+1, axis=0)
					class_dset[row, :] = meta_data

					row = uuid_dset.shape[0]
					class_dset.resize(row+1, axis=0)
					class_dset[row, :] = seriesuid_str

if __name__ == '__main__':
	main()
