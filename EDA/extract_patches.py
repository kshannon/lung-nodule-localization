#! /usr/bin/env python

# Thanks to Jonathan Mulholland and Aaron Sander from Booz Allen Hamilton who
# made their code publically availble, parts of which we are using in this script.
# https://www.kaggle.com/c/data-science-bowl-2017/details/tutorial

#### ---- Imports & Dependencies ---- ####
import sys
import os
import argparse
import pathlib
from glob import glob
from random import shuffle
import SimpleITK as sitk   # pip install SimpleITK
from tqdm import tqdm    # pip install tqdm
import h5py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.misc import imsave # might require: conda install Pillow


#### ---- Argparse Utility ---- ####
parser = argparse.ArgumentParser(description='Specify patch dimmensions,\
								data dir e.g. /home/data/,\
								the LUNA subset num,\
								and T/F to save pngs',
								add_help=True)
parser.add_argument('-dim', action="store", dest="dim", type=int, required=True)
parser.add_argument('-data', action="store", dest="data", type=str, required=True)
parser.add_argument('-subset', action="store", dest="subset", type=str, required=True)
parser.add_argument('-img', action="store_true", dest="img", default=False)
parser.add_argument('-tensor', action="store_true", dest="tensor", default=False)
args = parser.parse_args()


#### ---- Global Vars ---- ####
PATCH_DIM = args.dim
DATA_DIR = args.data
SUBSET = args.subset
SAVE_IMG = args.img
TENSOR = args.tensor
MASK_DIMS = tuple([int(PATCH_DIM/2)])*3 #set the width, height, depth, pass to make_mask()
FILE_LIST = glob("{}{}/*.mhd".format(DATA_DIR, SUBSET)) #list of .mhd files
	# FILE_LIST = glob("{}subset{}/*.mhd".format(DATA_DIR, SUBSET)
DF_NODE = pd.read_csv(DATA_DIR + "csv-files/candidates_with_annotations.csv")
	# DF_NODE = pd.read_csv(DATA_DIR+"csv-files/candidates_V2.csv")


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


def make_mask(center,diam,z,width,height,depth,spacing,origin,
			  mask_width=MASK_DIMS[0],mask_height=MASK_DIMS[1],mask_depth=MASK_DIMS[2]):
	"""
	Center : centers of circles px -- list of coordinates x,y,z
	diam : diameters of circles px -- diameter
	z = z position of slice in world coordinates mm
	width X height : pixel dim of image
	spacing = mm/px conversion rate np array x,y,z
	origin = x,y,z mm np.array
	"""
	mask = np.zeros([height,width]) # 0"s everywhere except nodule swapping x,y to match img
	#convert to nodule space from world coordinates

	padMask = 5

	# Defining the voxel range in which the nodule falls
	v_center = (center-origin)/spacing
	v_diam = int(diam/spacing[0]+padMask)
	v_xmin = np.max([0,int(v_center[0]-v_diam)-padMask])
	v_xmax = np.min([width-1,int(v_center[0]+v_diam)+padMask])
	v_ymin = np.max([0,int(v_center[1]-v_diam)-padMask])
	v_ymax = np.min([height-1,int(v_center[1]+v_diam)+padMask])

	v_xrange = range(v_xmin,v_xmax+1)
	v_yrange = range(v_ymin,v_ymax+1)

	# Convert back to world coordinates for distance calculation
	x_data = [x*spacing[0]+origin[0] for x in range(width)]
	y_data = [x*spacing[1]+origin[1] for x in range(height)]


	# RECTANGULAR MASK
	for v_x in v_xrange:
		for v_y in v_yrange:
			p_x = spacing[0]*v_x + origin[0]
			p_y = spacing[1]*v_y + origin[1]
			if ((p_x >= (center[0] - mask_width)) &
				(p_x <= (center[0] + mask_width)) &
				(p_y >= (center[1] - mask_height)) &
				(p_y <= (center[1] + mask_height))):

				mask[int((np.abs(p_y-origin[1]))/spacing[1]),
					int((np.abs(p_x-origin[0]))/spacing[0])] = 1.0


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

	print(type(bbox))
	print(bbox)

	return mask, bbox


def main():

	with h5py.File(DATA_DIR + str(PATCH_DIM) + 'dim_patches.hdf5', 'a') as HDF5:
		img_dset = HDF5.create_dataset('patches', (1,PATCH_DIM*PATCH_DIM), maxshape=(None,PATCH_DIM*PATCH_DIM))
		img_dset.attrs['patch_size'] = PATCH_DIM
		img_dset.attrs['plane'] = 'transverse'
		class_dset = HDF5.create_dataset('classes', (1,1), maxshape=(None,1), dtype=int)
		class_dset.attrs['classes'] = '2'
		print("Created HDF5 File and Datasets")

		for img_count, img_file in enumerate(tqdm(FILE_LIST)):

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

			for candidate_idx, cur_row in mini_df.iterrows(): # Iterate through all candidates

				# This is the real world x,y,z coordinates of possible nodule (in mm)
				candidate_x = cur_row["coordX"]
				candidate_y = cur_row["coordY"]
				candidate_z = cur_row["coordZ"]
				diam = cur_row["diameter_mm"]  # Only defined for true positives
				if np.isnan(diam):
					diam = 30.0  # If NaN, then just use a default of 30 mm

				class_id = cur_row["class"] #0 for false, 1 for true nodule

				mask_width = 32 # This is really the half width so window will be double this width
				mask_height = 32 # This is really the half height so window will be double this height
				mask_depth = 2 # This is really the half depth so window will be double this depth

				center = np.array([candidate_x, candidate_y, candidate_z])   # candidate center
				voxel_center = np.rint((center-origin)/spacing).astype(int)  # candidate center in voxel space (still x,y,z ordering)

				# Calculates the bounding box (and ROI mask) for desired position
				mask, bbox = make_mask(center, diam, voxel_center[2]*spacing[2]+origin[2],
									   width, height, slice_z, spacing, origin,
									   mask_width, mask_height, mask_depth)

				# a numpy array size of DIM x DIM
				# Confer with https://en.wikipedia.org/wiki/Anatomical_terms_of_location#Planes
				# Transverse slice 2D view - Y-X plane
				# img = img_array[bbox[2][0]:bbox[2][1],
				# 		bbox[0][0]:bbox[0][1],
				# 		bbox[1][0]:bbox[1][1]]
				# print(img.shape)

				if TENSOR:
					img_transverse = img_array[bbox[2][0]:bbox[2][1],
	                        bbox[0][0]:bbox[0][1],
	                        bbox[1][0]:bbox[1][1]]
					print(img.shape) #(60,64,64) mask_depth roughly half of this value [60]
				else:
					img_transverse = img_array[voxel_center[2],
						bbox[0][0]:bbox[0][1],
						bbox[1][0]:bbox[1][1]]
					print(img_transverse.shape) #(64,64)

				# sys.exit()


				# If -img argument passed will save the patch as a .png
				if SAVE_IMG:
					imsave(DATA_DIR + "sample_patches/class_{}_uid_{}_xyz_{}_{}_{}.png".format(
							class_id,
							seriesuid,
							candidate_x,
							candidate_y,
							candidate_z), img_transverse)

				# For now we will ignore imgs where the patch is getting clipped by the edge(s)
				img_transverse = normalizePlanes(img_transverse) #normalize HU units
				img_transverse = img_transverse.ravel().reshape(1,-1) #flatten img
				if img_transverse.shape[1] != PATCH_DIM * PATCH_DIM:
					continue
				if img_count == 0:
					img_dset[:] = img_transverse
					class_dset[:] = class_id
				else:
					row = img_dset.shape[0] # How many rows in the dataset currently?
					img_dset.resize(row+1, axis=0) # Add one more row (i.e. new ROI)
					img_dset[row, :] = img_transverse

					row = class_dset.shape[0] # How many rows in the dataset currently?
					class_dset.resize(row+1, axis=0) # Add one more row (i.e. new ROI)
					class_dset[row, :] = int(class_id)


if __name__ == '__main__':
	main()

	############
	#
	# Getting list of image files
	# output_path = "./patches/"
	# train_path = output_path + "train/"
	# validation_path = output_path + "validation/"
	#
	#
	# # Create the output directories if they don't exist
	# pathlib.Path(train_path+"class_0/").mkdir(parents=True, exist_ok=True)
	# pathlib.Path(train_path+"class_1/").mkdir(parents=True, exist_ok=True)
	# pathlib.Path(validation_path+"class_0/").mkdir(parents=True, exist_ok=True)
	# pathlib.Path(validation_path+"class_1/").mkdir(parents=True, exist_ok=True)


#### ---- Optional Side Quests ---- ####

# Sagittal slice 2D view - Z-Y plane
# img_sagittal = normalizePlanes(img_array[bbox[2][0]:bbox[2][1],
# 	bbox[0][0]:bbox[0][1],
# 	voxel_center[0]])

# Coronal slice 2D view - Z-X plane
# img_coronal = normalizePlanes(img_array[bbox[2][0]:bbox[2][1],
# 	voxel_center[1],
# 	bbox[1][0]:bbox[1][1]])

# SPHERICAL MASK
# Fill in 1 within sphere around nodule
#     for v_x in v_xrange:
#         for v_y in v_yrange:
#             p_x = spacing[0]*v_x + origin[0]
#             p_y = spacing[1]*v_y + origin[1]
#             if np.linalg.norm(center-np.array([p_x,p_y,z]))<=diam:
#                 mask[int((p_y-origin[1])/spacing[1]),int((p_x-origin[0])/spacing[0])] = 1.0
