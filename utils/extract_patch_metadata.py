#! /usr/bin/env python

#### ---- Imports & Dependencies ---- ####
import sys
import os
import argparse
import csv
from configparser import ConfigParser
import pathlib
from glob import glob
from random import shuffle
import SimpleITK as sitk # pip install SimpleITK
from tqdm import tqdm # pip install tqdm
import pandas as pd
import numpy as np
from scipy.misc import imsave # conda install Pillow or PIL


#### ---- Argparse Utility ---- ####
parser = argparse.ArgumentParser(description='Modify the patch metadata script',add_help=True)
parser.add_argument('-img',
					action="store_true",
					dest="img",
					default=False,
					help='Save .png patches to ./patches/')
parser.add_argument('-csv',
					action="store_true",
					dest="csv",
					default=True,
					help='Save processed data metadata to csv')
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
config.read('extract_patch_metadata_config.ini') #local just for now (need if - else for AWS)

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
SAVE_CSV = args.csv
PATCH_DIM = args.dim
NUM_SLICES = args.slices
# WORK_REMOTE = args.remote #add later w/ AWS
DF_NODE = pd.read_csv(CSV_PATH + "candidates_with_annotations.csv")
FILE_LIST = []
for unique_set in SUBSET:
	FILE_LIST.extend(glob("{}{}/*.mhd".format(LUNA_PATH, unique_set))) #add subset of .mhd files


#### ---- Helper Functions ---- ####
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


def write_to_csv(csvwriter,uuid,class_id,origin,center,diam,ct_img_diam):
	"""Accept patch processed data to write in CSV"""
	csvwriter.writerow([uuid,
						class_id,
						origin[0],
						origin[1],
						origin[2],
						center[0],
						center[1],
						center[2],
						diam,
						ct_img_diam[0],
						ct_img_diam[1],
						ct_img_diam[2]])
	return csvwriter


#### ---- Process CT Scans and extract Patches (the pipeline) ---- ####
def main():
	"""
	Create the csv file + meta data, iterate thriough the folders DICOM imgs
	Normalize the imgs, create mini patches and write them to the hdf5 file system
	"""
	filename = CSV_PATH + "patch_metadata.csv"
	with open(filename, "w") as csvfile:
		csvwriter = csv.writer(csvfile,  delimiter=',')
		csvwriter.writerow(["seriesuid",
							"class_id",
							"origin_x",
							"origin_y",
							"origin_z",
							"center_x",
							"center_y",
							"center_z",
							"diam",
							"ct_dim_x",
							"ct_dim_y",
							"ct_dim_z"])

		#### ---- Iterating through a CT scan ---- ####
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
			ct_img_dim = img_array.shape
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

				write_to_csv(csvwriter,seriesuid,class_id,origin,center,diam,ct_img_dim) #write 12 cols.


	print("All Images Processed and Metadata written to csv. Thank you patch again!")
	print('\a')

if __name__ == '__main__':
	main()
