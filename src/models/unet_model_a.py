"""
batch norm: False
HU norm: False
learning rate: .001
weighted loss function: false
stratified sampling: 1:1 (+ : -)
resolution layers: 3
"""

# example run: python luna16_3dcomp_cnn_UNET.py --gpuid 0 --datadir ~/vol/hdf5/ --holdout 0 --batchsize 16
# python luna16_3dcomp_cnn_UNET.py --gpuid 0 --datadir ~/Users/keil/datasets/LUNA16/ --holdout 0 --batchsize 16
# use this statement:
# python luna16_3dcomp_cnn_UNET.py --gpuid 0 --datadir ~/datasets/LUNA16/ --holdout 0 --batchsize 16

import argparse
import sys
parser = argparse.ArgumentParser(description='Modify the training script',add_help=True)

# root_dir = !pwd
# s3bucket_path = root_dir[0] + '/../s3bucket_goofys/' # remote S3 via goofys
parser.add_argument("--gpuid", default=1, type=int, help="GPU to use (0-3)") # gpu id depends on machine class
parser.add_argument("--datadir", default="/nfs/site/home/ganthony/", help="Path to data hdf5 file")
parser.add_argument("--holdout", type=int, default=0, help="subset of data to skip during training.")
parser.add_argument("--batchsize", type=int, default=16, help="The batch size to use")
args = parser.parse_args()

import os
os.environ["CUDA_VISIBLE_DEVICES"]="{}".format(args.gpuid)  # Only use gpu #1 (0-4)

data_dir = args.datadir
HOLDOUT_SUBSET = args.holdout

path_to_hdf5 = data_dir + "64x64x64-patch-withdiam.hdf5"
#path_to_hdf5 = data_dir + "64x64x64-patch-annotations.hdf5"

TB_LOG_DIR = "../logs/tb_3D_unet_logs"

crop_shape = (48,48,48,1) # Change to correct crop shape

import time
# Save Keras model to this file
CHECKPOINT_FILENAME = "./cnn_3DUNET_modelA_64_64_64_HOLDOUT{}".format(HOLDOUT_SUBSET) + time.strftime("_%Y%m%d_%H%M%S") + ".hdf5"

print(CHECKPOINT_FILENAME)

import tensorflow as tf

import keras
config = tf.ConfigProto()
config.gpu_options.allow_growth=True # Don't use all GPU memory if not needed
sess = tf.Session(config=config)
keras.backend.set_session(sess)

import numpy as np
import h5py
import os
import math

def dice_coef(target, prediction, axis=(1, 2, 3), smooth=1e-5):
	"""
	Sorenson Dice
	"""
	intersection = tf.reduce_sum(prediction * target, axis=axis)
	p = tf.reduce_sum(prediction, axis=axis)
	t = tf.reduce_sum(target, axis=axis)
	dice = (2. * intersection + smooth) / (t + p + smooth)

	return tf.reduce_mean(dice)

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

CHANNEL_LAST = True
if CHANNEL_LAST:
	concat_axis = -1
	data_format = "channels_last"

else:
	concat_axis = 1
	data_format = "channels_first"


def unet3D(input_img, use_upsampling=False, n_out=1, dropout=0.2,
			print_summary = False):
	"""
	3D U-Net model
	"""
	print("3D U-Net Segmentation")
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

# Branch checkpointer
from keras.callbacks import Callback

class WeightsSaver(Callback):
	def __init__(self, model, N, fname):
		self.model = model
		self.N = N
		self.batch = 0
		self.fname = fname

	def on_batch_end(self, batch, logs={}):
		if self.batch % self.N == 0:
			self.model.save_weights(self.fname+'weights.h5')
			self.model.save(self.fname +'checkpoint.h5')
		self.batch += 1

patch_dim = 64
max_diam = 33 # this need to be changed to general data specific

# creating mask dictionary
masks = {}
max_radius = math.ceil(max_diam/2)
for radius in range(max_radius):
	mask = np.zeros((patch_dim,patch_dim,patch_dim,1))
	if radius > 0:
		for i in range(patch_dim):
			for j in range(patch_dim):
				for k in range(patch_dim):
					half = patch_dim/2
					if (np.sqrt((i-half)**2+(j-half)**2+(k-half)**2) <=radius):
						mask[i,j,k]=1
	masks[radius]=mask


def get_class_idx(hdf5_file, classid = 0):
	'''
	Get the indices for the class classid and valid for training
	'''
#     # 1. Find indices from class classid
#     idx_class = np.where( (hdf5_file['output'][:,0] == classid) )[0]

#     # 2. Find indices that are not excluded from training
#     idx_notraining = np.where(hdf5_file["notrain"][:,0] == 1)[0]

#     # 1. Find indices from class classid
#     idx_class = np.where( (hdf5_file['output'][:,0] == classid) )[0]

#     # 2. Find indices that are not excluded from training
#     idx_notraining = np.where(hdf5_file["notrain"][:,0] == 1)[0]

	 # 1. Find indices from class classid
	idx_class = np.where( (hdf5_file['output'][:,0] == classid) )[0]

	# 2. Find indices that are not excluded from training
	idx_notraining = np.where(hdf5_file["notrain"][:,0] == 1)[0]

	print("get_class_idx")
	return np.setdiff1d(idx_class, idx_notraining)

def remove_exclude_subset_idx(hdf5_file, idx, excluded_subset=0):
	'''
	Remove indices for the subset excluded_subset
	'''

	excluded_idx = np.where(hdf5_file["subsets"][:,0] == excluded_subset)[0] # indices

	return np.setdiff1d(idx, excluded_idx)  # Remove the indices of the excluded subset

def get_idx_for_classes(hdf5_file, excluded_subset=0):
	'''
	Get the indices for each class but don't include indices from excluded subset
	'''

	idx = {}
	idx[0] = get_class_idx(hdf5_file, 0)
	idx[1] = get_class_idx(hdf5_file, 1)

	idx[0] = remove_exclude_subset_idx(hdf5_file, idx[0], excluded_subset)
	idx[1] = remove_exclude_subset_idx(hdf5_file, idx[1], excluded_subset)

	return idx

def get_random_idx(idx, batch_size = 20):
	'''
	Batch size needs to be even.
	This is yield a balanced set of random indices for each class.
	'''

	idx0 = idx[0]
	idx1 = idx[1]

	# 2. Shuffle the two indices
	np.random.shuffle(idx0)  # This shuffles in place
	np.random.shuffle(idx1)  # This shuffles in place

	# 3. Take half of the batch from each class
	idx0_shuffle = idx0[0:(batch_size//2)]
	idx1_shuffle = idx1[0:(batch_size//2)]

	# Need to sort final list in order to slice
	return np.sort(np.append(idx0_shuffle, idx1_shuffle))

def img_rotate(img, msk):
	'''
	Perform a random rotation on the tensor
	`img` is the tensor and `msk` is the mask
	'''
	shape = img.shape

	if (shape[0] == shape[1]) & (shape[1] == shape[2]):
		same_dims = 3
	elif (shape[0] == shape[1]):
		same_dims = 2
	else:
		print("ERROR: Image should be square or cubed to flip")

	# This will flip along n-1 axes. (If we flipped all n axes then we'd get the same result every time)
	ax = np.random.choice(same_dims,len(shape)-2, replace=False) # Choose randomly which axes to rotate

	# The flip allows the negative/positive rotation
	amount_rot = np.random.permutation([-3,-2,-1,1,2,3])[0]
	return np.rot90(img, amount_rot, (ax[0], ax[1])),np.rot90(msk, amount_rot, (ax[0], ax[1]))# Random rotation

def img_flip(img, msk):
	'''
	Performs a random flip on the tensor.
	If the tensor is C x H x W x D this will perform flips on two of the C, H, D dimensions
	If the tensor is C x H x W this will perform flip on either the H or the W dimension.
	`img` is the tensor
	'''
	shape = img.shape
	flip_axis = np.random.permutation([0,1])[0]
	img = np.flip(img, flip_axis) # Flip along random axis
	msk = np.flip(msk, flip_axis)
	return img, msk

def augment_data(imgs, msks,validation = False):
	'''
	Performs random flips, rotations, and other operations on the image tensors.
	'''

	imgs_length = imgs.shape[0]
	imgs_crop = np.zeros((imgs_length, crop_shape[0],crop_shape[1],crop_shape[2],crop_shape[3]))
	msks_crop = np.zeros_like(imgs_crop)

	if not validation:
		for idx in range(imgs_length):
			img = imgs[idx, :]
			msk = msks[idx, :]

			if (np.random.rand() > 0.5):

				if (np.random.rand() > 0.5):
					img, msk = img_rotate(img, msk)

				if (np.random.rand() > 0.5):
					img, msk = img_flip(img, msk)

			else:

				if (np.random.rand() > 0.5):
					img, msk = img_flip(img, msk)

				if (np.random.rand() > 0.5):
					img, msk = img_rotate(img, msk)

			img, msk = crop_img(img, msk, False)

			imgs_crop[idx,:] = img
			msks_crop[idx, :] = msk

		else:
			for idx in range(imgs_length):
				img = imgs[idx, :]
				msk = msks[idx, :]
				img, msk = crop_img(img, msk, True)
				imgs_crop[idx,:] = img
				msks_crop[idx, :] = msk

	return imgs_crop, msks_crop

def crop_img(img, msk,valid_flag = False):
	"""
	Peforms random crop with offset
	"""
	if(valid_flag == False):
		offsetX = np.random.randint(0,64-crop_shape[0]-1)
		offsetY = np.random.randint(0,64-crop_shape[1]-1)
		offsetZ = np.random.randint(0,64-crop_shape[2]-1)
	else:
		offsetX = 8 # (64-48)/2
		offsetY = 8
		offsetZ = 8
	start_idxX = offsetX
	start_idxY = offsetY
	start_idxZ = offsetZ
	stop_idxX = crop_shape[0] + offsetX
	stop_idxY = crop_shape[1] + offsetY
	stop_idxZ = crop_shape[2] + offsetZ
	img = img[start_idxX:stop_idxX, start_idxY:stop_idxY, start_idxZ:stop_idxZ, :]
	msk = msk[start_idxX:stop_idxX, start_idxY:stop_idxY, start_idxZ:stop_idxZ, :]

	return img, msk
def get_batch(hdf5_file, batch_size=50, exclude_subset=0):
	"""Replaces Keras' native ImageDataGenerator."""
	""" Randomly select batch_size rows from the hdf5 file dataset """

	#input_shape = tuple([batch_size] + list(hdf5_file['input'].attrs['lshape']) + [1])
	input_shape = (batch_size, 64,64,64,1)

	idx_master = get_idx_for_classes(hdf5_file, exclude_subset)

	random_idx = get_random_idx(idx_master, batch_size)
	imgs = hdf5_file["input"][random_idx,:]

	imgs = imgs.reshape(input_shape)
	imgs = np.swapaxes(imgs, 1,3)

	# diameters is a vector with the diameters of the nodules in the batch
	# For class 0, pass a diameter of 0.
	#msks = create_mask(imgs.shape, diameters)

	# until the code for masking is addressed, take zeros for
	#msks = np.zeros_like(imgs)
	list_msks = []
	for i in random_idx:
		diam_patch = hdf5_file["diameter_label"][i, 0]
		list_msks.append(masks[math.ceil(diam_patch/2)])
	msks = np.array(list_msks)

	#classes = hdf5_file["output"][random_idx, 0]

	return imgs, msks

def generate_data(hdf5_file, batch_size=50, subset=0, validation=False):
	"""Replaces Keras' native ImageDataGenerator."""
	""" Randomly select batch_size rows from the hdf5 file dataset """

	# If validation, then get the subset
	# If not validation (training), then get everything but the subset.
	if validation:
		idx_master = get_idx_for_onesubset(hdf5_file, subset)

	else:
		idx_master = get_idx_for_classes(hdf5_file, subset)

	#input_shape = tuple([batch_size] + list(hdf5_file['input'].attrs['lshape']) + [1])
	input_shape = (batch_size, 64,64,64,1) # check this input shape for 3D also check if above can be taken out and replace with this line

	while True:

		random_idx = get_random_idx(idx_master, batch_size)
		imgs = hdf5_file["input"][random_idx,:]
		imgs = imgs.reshape(input_shape)
		imgs = np.swapaxes(imgs, 1, 3)
		#msks= np.zeros_like(imgs)
		list_msks = []
		for i in random_idx:
			diam_patch = hdf5_file["diameter_label"][i, 0]
			list_msks.append(masks[math.ceil(diam_patch/2)])
		msks = np.array(list_msks)

		# classes = hdf5_file["output"][random_idx, 0]

		# Normalization to HO units

		for i in range(len(random_idx)):
			temp_img  = imgs[i,:,:,:,0]
			imgs[i,:,:,:,0] = normalize(temp_img)

		if not validation:  # Training need augmentation. Validation does not.
			## Need to augment
			imgs, msks = augment_data(imgs, msks, False)

		else: # but requires cropping to be of same size as model expects input of 48x48x48
			imgs, msks = augment_data(imgs, msks, True)

		#classes = hdf5_file["output"][random_idx, 0]

		yield [imgs], [msks] #, classes]

def normalize(img):
	# maxHU, minHU = 400., -1000.
	# img = (img - minHU) / (maxHU - minHU)
	# img[img>1] = 1.
	# img[img<0] = 0.
	return img


def get_idx_for_onesubset(hdf5_file, subset=0):
	'''
	Get the indices for one subset to be used in testing/validation
	'''

	idx_subset = np.where( (hdf5_file["subsets"][:,0] == subset) )[0]

	idx = {}
	idx[0] = np.where( (hdf5_file['output'][idx_subset,0] == 0) )[0]
	idx[1] = np.where( (hdf5_file['output'][idx_subset,0] == 1) )[0]

	return idx

#### MAIN  ######

with h5py.File(path_to_hdf5, 'r') as hdf5_file: # open in read-only mode

	print("Valid hdf5 file in 'read' mode: " + str(hdf5_file))
	file_size = os.path.getsize(path_to_hdf5)
	print('Size of hdf5 file: {:.3f} GB'.format(file_size/2.0**30))

	num_rows = hdf5_file['input'].shape[0]
	print("There are {} images in the dataset.".format(num_rows))

	print("The datasets within the HDF5 file are:\n {}".format(list(hdf5_file.values())))

	input_shape = tuple(list(hdf5_file["input"].attrs["lshape"]))
	batch_size = args.batchsize   # Batch size to use
	print ("Input shape of tensor = {}".format(input_shape))

	#from resnet3d import Resnet3DBuilder

	#model = Resnet3DBuilder.build_resnet_18((64, 64, 64, 1), 1)  # (input tensor shape, number of outputs)
	model = unet3D(crop_shape,use_upsampling=True,n_out=1,dropout=0.2,print_summary=True)

	tb_log = keras.callbacks.TensorBoard(log_dir=TB_LOG_DIR,
								histogram_freq=0,
								batch_size=batch_size,
								write_graph=True,
								write_grads=True,
								write_images=True,
								embeddings_freq=0,
								embeddings_layer_names=None,
								embeddings_metadata=None)
	#print(model.metrics_names)
	checkpointer = keras.callbacks.ModelCheckpoint(filepath=CHECKPOINT_FILENAME,
												   #monitor ="val_PredictionMask_loss",
												   verbose=1,
												   save_best_only=True)
	print(type(checkpointer))
	# model.compile(optimizer='sgd', #'adam',
	#               loss='binary_crossentropy',
	#               metrics=['accuracy'])

	model.compile(optimizer='adam',
				  #loss=[dice_coef_loss],#pred_msk,class_pred
				  #loss={'main_output': 'binary_crossentropy', 'aux_output': 'binary_crossentropy'}, loss_weights={'main_output': 1., 'aux_output': 0.2})
				  #loss=[dice_coef_loss,'binary_crossentropy'],
				  # loss={'PredictionMask': dice_coef_loss}, loss_weights={'PredictionMask': 1.0},
				  # metrics={'PredictionMask':dice_coef})
				  loss=[dice_coef_loss],
				  metrics= [dice_coef])

	# print(model.summary())

	#validation_batch_size = 32
	train_generator = generate_data(hdf5_file, batch_size, subset=HOLDOUT_SUBSET, validation=False)
	validation_generator = generate_data(hdf5_file, batch_size, subset=HOLDOUT_SUBSET, validation=True)

	history = model.fit_generator(train_generator,
						#steps_per_epoch=1,epochs=1, # this needs to be changed to back to below actual steps per epoch and origninal epochs
						steps_per_epoch=num_rows//batch_size, epochs=1,
						validation_data = validation_generator,
						validation_steps = 100,
						callbacks=[tb_log, checkpointer, WeightsSaver(model, 5000, CHECKPOINT_FILENAME)])

	# save as JSON and saving model weights
	UNET_json_arch = model.to_json()
	model.save('UNET_modelA_H{}.h5'.format(HOLDOUT_SUBSET))
	model.save_weights('UNET_weights_modelA_H{}.h5'.format(HOLDOUT_SUBSET))
