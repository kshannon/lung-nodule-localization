import pandas as pd
import numpy as np
from keras.models import load_model
import h5py

import pandas as pd
import argparse
import tensorflow as tf
import SimpleITK as sitk
import matplotlib.pyplot as plt
from PIL import Image
import os

import tensorflow as tf
import keras

def dice_coef_loss(target, prediction, axis=(1,2,3), smooth=1e-5):
	"""
	Sorenson Dice loss
	Using -log(Dice) as the loss since it is better behaved.
	Also, the log allows avoidance of the division which
	can help prevent underflow when the numbers are very small.
	"""
	intersection = tf.reduce_sum(prediction * target, axis=axis)
	p = tf.reduce_sum(prediction, axis=axis)
	t = tf.reduce_sum(target, axis=axis)
	numerator = tf.reduce_mean(2. * intersection + smooth)
	denominator = tf.reduce_mean(t + p + smooth)
	dice_loss = -tf.log(numerator) + tf.log(denominator)
	return dice_loss

def dice_coef(target, prediction, axis=(1, 2, 3), smooth=1e-5):
	"""
	Sorenson Dice
	"""
	intersection = tf.reduce_sum(prediction * target, axis=axis)
	p = tf.reduce_sum(prediction, axis=axis)
	t = tf.reduce_sum(target, axis=axis)
	dice = (2. * intersection + smooth) / (t + p + smooth)

	return tf.reduce_mean(dice)

def normalize_HU(img):
	maxHU, minHU = 400., -1000.
	img = (img - minHU) / (maxHU - minHU)
	img[img>1] = 1.
	img[img<0] = 0.
	return img

def normalize_img(img):
	pixel_spacing = [1.0, 1.0, 1.0]  # New Voxel spacing in mm
	new_x_size = img.GetSpacing()[0]*img.GetWidth()  # Number of Voxels you want for x dimension
	new_y_size = img.GetSpacing()[1]*img.GetHeight() # Number of Voxels you want for y dimension
	new_z_size = img.GetSpacing()[2]*img.GetDepth()  # Number of Voxels you want for z dimesion
	new_size = [new_x_size, new_y_size, new_z_size]

	new_spacing = pixel_spacing  # mm per voxel (x,y,z) (h, w, d)
	new_size = np.rint(np.array(new_size) / np.array(new_spacing)).astype(int)
	interpolator_type = sitk.sitkBSpline
	img_norm = sitk.Resample(img, np.array(new_size, dtype='uint32').tolist(), sitk.Transform(), interpolator_type, img.GetOrigin(),                             new_spacing, img.GetDirection(), 0.0, img.GetPixelIDValue())
	img_norm.SetOrigin(np.array(img.GetOrigin()) / np.array(new_spacing))

	return img_norm

def create_unet3D_Model_A(input_img, use_upsampling=False, n_out=1, dropout=0.2,
			print_summary = False):
	"""
	3D U-Net model - Model-A
	"""
	concat_axis = -1
	data_format = "channels_last"
	# print("3D U-Net Segmentation")
	# Set keras learning phase to train
	keras.backend.set_learning_phase(True)

	# Don"t initialize variables on the fly
	keras.backend.manual_variable_initialization(False)

	inputs = keras.layers.Input(shape=input_img, name="Input_Image")

	# Use below if wanted to use batch normalization and Relu activation separately
	params = dict(kernel_size=(3, 3, 3), activation=None,
				  padding="same", data_format=data_format,
				  kernel_initializer="he_uniform")

	# params = dict(kernel_size=(3, 3, 3), activation="relu",
	# 			  padding="same", data_format=data_format,
	# 			  kernel_initializer="he_uniform")

	conv1 = keras.layers.Conv3D(name="conv1a", filters=32, **params)(inputs)
	# conv1 = keras.layers.BatchNormalization(axis =-1)(conv1)
	conv1 = keras.layers.Activation('relu')(conv1)
	conv1 = keras.layers.Conv3D(name="conv1b", filters=64, **params)(conv1)
	# conv1 = keras.layers.BatchNormalization(axis =-1)(conv1)
	conv1 = keras.layers.Activation('relu')(conv1)
	pool1 = keras.layers.MaxPooling3D(name="pool1", pool_size=(2, 2, 2))(conv1)

	conv2 = keras.layers.Conv3D(name="conv2a", filters=64, **params)(pool1)
	# conv2 = keras.layers.BatchNormalization(axis =-1)(conv2)
	conv2 = keras.layers.Activation('relu')(conv2)
	conv2 = keras.layers.Conv3D(name="conv2b", filters=128, **params)(conv2)
	# conv2 = keras.layers.BatchNormalization(axis =-1)(conv2)
	conv2 = keras.layers.Activation('relu')(conv2)
	pool2 = keras.layers.MaxPooling3D(name="pool2", pool_size=(2, 2, 2))(conv2)

	conv3 = keras.layers.Conv3D(name="conv3a", filters=128, **params)(pool2)
	# conv3 = keras.layers.BatchNormalization(axis =-1)(conv3)
	conv3 = keras.layers.Activation('relu')(conv3)
	conv3 = keras.layers.Dropout(dropout)(conv3) ### Trying dropout layers earlier on, as indicated in the paper
	conv3 = keras.layers.Conv3D(name="conv3b", filters=256, **params)(conv3)
	# conv3 = keras.layers.BatchNormalization(axis =-1)(conv3)
	conv3 = keras.layers.Activation('relu')(conv3)

	if use_upsampling:
		up3 = keras.layers.concatenate([keras.layers.UpSampling3D(name="up3", size=(2, 2, 2))(conv3), conv2], axis=concat_axis)
	else:
		up3 = keras.layers.concatenate([keras.layers.Conv3DTranspose(name="transConv3", filters=256, data_format=data_format,
						   kernel_size=(2, 2, 2), strides=(2, 2, 2), padding="same")(conv3), conv2], axis=concat_axis)


	conv4 = keras.layers.Conv3D(name="conv4a", filters=128, **params)(up3)
	conv4 = keras.layers.Activation('relu')(conv4)
	conv4 = keras.layers.Conv3D(name="conv4b", filters=128, **params)(conv4)
	conv4 = keras.layers.Activation('relu')(conv4)

	if use_upsampling:
		up4 = keras.layers.concatenate([keras.layers.UpSampling3D(name="up4", size=(2, 2, 2))(conv4), conv1], axis=concat_axis)
	else:
		up4 = keras.layers.concatenate([keras.layers.Conv3DTranspose(name="transConv4", filters=128, data_format=data_format,
						   kernel_size=(2, 2, 2), strides=(2, 2, 2), padding="same")(conv4), conv1], axis=concat_axis)

	conv5 = keras.layers.Conv3D(name="conv5a", filters=64, **params)(up4)
	conv5 = keras.layers.Activation('relu')(conv5)
	conv5 = keras.layers.Conv3D(name="conv5b", filters=32, **params)(conv5)
	conv5 = keras.layers.Activation('relu')(conv5)


	pred_msk = keras.layers.Conv3D(name="PredictionMask", filters=n_out, kernel_size=(1, 1, 1),
					data_format=data_format, activation="sigmoid")(conv5)

	model = keras.models.Model(inputs=[inputs], outputs=[pred_msk])

	if print_summary:
		#model = keras.models.Model(inputs=[inputs], outputs=[class_pred])
		model.summary()

	# return pred
	return model

def create_UNET3D(input_img, use_upsampling=False, n_out=1, dropout=0.2,
			print_summary = False):
	"""
	3D U-Net model architecture creation
	"""

	concat_axis = -1
	data_format = "channels_last"
	keras.backend.set_learning_phase(True)

	# Don"t initialize variables on the fly
	keras.backend.manual_variable_initialization(False)

	inputs = keras.layers.Input(shape=input_img, name="Input_Image")

	# Use below if wanted to use batch normalization and Relu activation separately
	params = dict(kernel_size=(3, 3, 3), activation=None,
				  padding="same", data_format=data_format,
				  kernel_initializer="he_uniform")

	conv1 = keras.layers.Conv3D(name="conv1a", filters=32, **params)(inputs)
	conv1 = keras.layers.BatchNormalization(axis =-1)(conv1)
	conv1 = keras.layers.Activation('relu')(conv1)
	conv1 = keras.layers.Conv3D(name="conv1b", filters=64, **params)(conv1)
	conv1 = keras.layers.BatchNormalization(axis =-1)(conv1)
	conv1 = keras.layers.Activation('relu')(conv1)
	pool1 = keras.layers.MaxPooling3D(name="pool1", pool_size=(2, 2, 2))(conv1)

	conv2 = keras.layers.Conv3D(name="conv2a", filters=64, **params)(pool1)
	conv2 = keras.layers.BatchNormalization(axis =-1)(conv2)
	conv2 = keras.layers.Activation('relu')(conv2)
	conv2 = keras.layers.Conv3D(name="conv2b", filters=128, **params)(conv2)
	conv2 = keras.layers.BatchNormalization(axis =-1)(conv2)
	conv2 = keras.layers.Activation('relu')(conv2)
	pool2 = keras.layers.MaxPooling3D(name="pool2", pool_size=(2, 2, 2))(conv2)

	conv3 = keras.layers.Conv3D(name="conv3a", filters=128, **params)(pool2)
	conv3 = keras.layers.BatchNormalization(axis =-1)(conv3)
	conv3 = keras.layers.Activation('relu')(conv3)
	conv3 = keras.layers.Dropout(dropout)(conv3)
	conv3 = keras.layers.Conv3D(name="conv3b", filters=256, **params)(conv3)
	conv3 = keras.layers.BatchNormalization(axis =-1)(conv3)
	conv3 = keras.layers.Activation('relu')(conv3)
	pool3 = keras.layers.MaxPooling3D(name="pool3", pool_size=(2, 2, 2))(conv3)

	conv4 = keras.layers.Conv3D(name="conv4a", filters=256, **params)(pool3)
	conv4 = keras.layers.BatchNormalization(axis =-1)(conv4)
	conv4 = keras.layers.Activation('relu')(conv4)
	conv4 = keras.layers.Dropout(dropout)(conv4)
	conv4 = keras.layers.Conv3D(name="conv4b", filters=512, **params)(conv4)
	conv4 = keras.layers.BatchNormalization(axis =-1)(conv4)
	conv4 = keras.layers.Activation('relu')(conv4)

	if use_upsampling:
		up4 = keras.layers.concatenate([keras.layers.UpSampling3D(name="up4", size=(2, 2, 2))(conv4), conv3], axis=concat_axis)
	else:
		up4 = keras.layers.concatenate([keras.layers.Conv3DTranspose(name="transConv4", filters=512, data_format=data_format,
						   kernel_size=(2, 2, 2), strides=(2, 2, 2), padding="same")(conv4), conv3], axis=concat_axis)


	conv5 = keras.layers.Conv3D(name="conv5a", filters=256, **params)(up4)
	conv5 = keras.layers.BatchNormalization(axis =-1)(conv5)
	conv5 = keras.layers.Activation('relu')(conv5)
	conv5 = keras.layers.Conv3D(name="conv5b", filters=256, **params)(conv5)
	conv5 = keras.layers.BatchNormalization(axis =-1)(conv5)
	conv5 = keras.layers.Activation('relu')(conv5)

	if use_upsampling:
		up5 = keras.layers.concatenate([keras.layers.UpSampling3D(name="up5", size=(2, 2, 2))(conv5), conv2], axis=concat_axis)
	else:
		up5 = keras.layers.concatenate([keras.layers.Conv3DTranspose(name="transConv5", filters=256, data_format=data_format,
						   kernel_size=(2, 2, 2), strides=(2, 2, 2), padding="same")(conv5), conv2], axis=concat_axis)

	conv6 = keras.layers.Conv3D(name="conv6a", filters=128, **params)(up5)
	conv6 = keras.layers.BatchNormalization(axis =-1)(conv6)
	conv6 = keras.layers.Activation('relu')(conv6)
	conv6 = keras.layers.Conv3D(name="conv6b", filters=128, **params)(conv6)
	conv6 = keras.layers.BatchNormalization(axis =-1)(conv6)
	conv6 = keras.layers.Activation('relu')(conv6)

	if use_upsampling:
		up6 = keras.layers.concatenate([keras.layers.UpSampling3D(name="up6", size=(2, 2, 2))(conv6), conv1], axis=concat_axis)
	else:
		up6 = keras.layers.concatenate([keras.layers.Conv3DTranspose(name="transConv6", filters=128, data_format=data_format,
						   kernel_size=(2, 2, 2), strides=(2, 2, 2), padding="same")(conv6), conv1], axis=concat_axis)

	conv7 = keras.layers.Conv3D(name="conv7a", filters=128, **params)(up6)
	conv7 = keras.layers.BatchNormalization(axis =-1)(conv7)
	conv7 = keras.layers.Activation('relu')(conv7)
	conv7 = keras.layers.Conv3D(name="conv7b", filters=128, **params)(conv7)
	conv7 = keras.layers.BatchNormalization(axis =-1)(conv7)
	conv7 = keras.layers.Activation('relu')(conv7)

	pred_msk = keras.layers.Conv3D(name="PredictionMask", filters=n_out, kernel_size=(1, 1, 1),
					data_format=data_format, activation="sigmoid")(conv7)

	#Branch is created from conv7 which are feature maps
	#But global avg pooling on feature maps is not helping and hence changing back to pred_msk
	class_pred = keras.layers.GlobalAveragePooling3D(name='PredictionClass')(pred_msk)
	model = keras.models.Model(inputs=[inputs], outputs=[pred_msk,class_pred])

	if print_summary:
		model.summary()

	return model
