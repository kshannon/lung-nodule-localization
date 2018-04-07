
import argparse
parser = argparse.ArgumentParser(description='Modify the training script',add_help=True)

# root_dir = !pwd
# s3bucket_path = root_dir[0] + '/../s3bucket_goofys/' # remote S3 via goofys
parser.add_argument("--gpuid", default=1, type=int, help="GPU to use (0-3)")
parser.add_argument("--datadir", default="/nfs/site/home/ganthony/", help="Path to data hdf5 file")
parser.add_argument("--holdout", type=int, default=0, help="subset of data to skip during training.")
parser.add_argument("--batchsize", type=int, default=512, help="The batch size to use")
args = parser.parse_args()

import os
os.environ["CUDA_VISIBLE_DEVICES"]="{}".format(args.gpuid)  # Only use gpu #1 (0-4)

data_dir = args.datadir
HOLDOUT_SUBSET = args.holdout

path_to_hdf5 = data_dir + "64x64x64-patch.hdf5"

TB_LOG_DIR = "./tb_3D_unet_logs"

crop_shape = (48,48,48,1) # Change to correct crop shape

print(path_to_hdf5)

import time
# Save Keras model to this file
CHECKPOINT_FILENAME = "./cnn_3d_64_64_64_HOLDOUT{}".format(HOLDOUT_SUBSET) + time.strftime("_%Y%m%d_%H%M%S") + ".hdf5"

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

	params = dict(kernel_size=(3, 3, 3), activation="relu",
				  padding="same", data_format=data_format,
				  kernel_initializer="he_uniform")

	conv1 = keras.layers.Conv3D(name="conv1a", filters=32, **params)(inputs)
	conv1 = keras.layers.Conv3D(name="conv1b", filters=64, **params)(conv1)
	pool1 = keras.layers.MaxPooling3D(name="pool1", pool_size=(2, 2, 2))(conv1)

	conv2 = keras.layers.Conv3D(name="conv2a", filters=64, **params)(pool1)
	conv2 = keras.layers.Conv3D(name="conv2b", filters=128, **params)(conv2)
	pool2 = keras.layers.MaxPooling3D(name="pool2", pool_size=(2, 2, 2))(conv2)

	conv3 = keras.layers.Conv3D(name="conv3a", filters=128, **params)(pool2)
	conv3 = keras.layers.Dropout(dropout)(conv3) ### Trying dropout layers earlier on, as indicated in the paper
	conv3 = keras.layers.Conv3D(name="conv3b", filters=256, **params)(conv3)
	pool3 = keras.layers.MaxPooling3D(name="pool3", pool_size=(2, 2, 2))(conv3)

	conv4 = keras.layers.Conv3D(name="conv4a", filters=256, **params)(pool3)
	conv4 = keras.layers.Dropout(dropout)(conv4) ### Trying dropout layers earlier on, as indicated in the paper
	conv4 = keras.layers.Conv3D(name="conv4b", filters=512, **params)(conv4)

	if use_upsampling:
		up4 = keras.layers.concatenate([keras.layers.UpSampling3D(name="up4", size=(2, 2, 2))(conv4), conv3], axis=concat_axis)
	else:
		up4 = keras.layers.concatenate([keras.layers.Conv3DTranspose(name="transConv4", filters=512, data_format=data_format,
						   kernel_size=(2, 2, 2), strides=(2, 2, 2), padding="same")(conv4), conv3], axis=concat_axis)


	conv5 = keras.layers.Conv3D(name="conv5a", filters=256, **params)(up4)
	conv5 = keras.layers.Conv3D(name="conv5b", filters=256, **params)(conv5)

	if use_upsampling:
		up5 = keras.layers.concatenate([keras.layers.UpSampling3D(name="up5", size=(2, 2, 2))(conv5), conv2], axis=concat_axis)
	else:
		up5 = keras.layers.concatenate([keras.layers.Conv3DTranspose(name="transConv5", filters=256, data_format=data_format,
						   kernel_size=(2, 2, 2), strides=(2, 2, 2), padding="same")(conv5), conv2], axis=concat_axis)

	conv6 = keras.layers.Conv3D(name="conv6a", filters=128, **params)(up5)
	conv6 = keras.layers.Conv3D(name="conv6b", filters=128, **params)(conv6)

	if use_upsampling:
		up6 = keras.layers.concatenate([keras.layers.UpSampling3D(name="up6", size=(2, 2, 2))(conv6), conv1], axis=concat_axis)
	else:
		up6 = keras.layers.concatenate([keras.layers.Conv3DTranspose(name="transConv6", filters=128, data_format=data_format,
						   kernel_size=(2, 2, 2), strides=(2, 2, 2), padding="same")(conv6), conv1], axis=concat_axis)

	conv7 = keras.layers.Conv3D(name="conv7a", filters=128, **params)(up6)
	conv7 = keras.layers.Conv3D(name="conv7b", filters=128, **params)(conv7)
	pred = keras.layers.Conv3D(name="PredictionMask", filters=n_out, kernel_size=(1, 1, 1),
					data_format=data_format, activation="sigmoid")(conv7)

	if print_summary:
		model = keras.models.Model(inputs=[inputs], outputs=[pred])
		model.summary()

	return pred

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
    return np.rot90(img, amount_rot, (ax[0], ax[1])),
            np.rot90(msk, amount_rot, (ax[0], ax[1]))# Random rotation

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

def augment_data(imgs, msks):
    '''
    Performs random flips, rotations, and other operations on the image tensors.
    '''

    imgs_length = imgs.shape[0]

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

        img, msk = crop_img(img, msk)

        imgs[idx,:] = img
        msks[idx, :] = msk

    return imgs, msks

def crop_img(img, msk):
    """
    Peforms random crop with offset
    """

    offsetX = np.rand.randint(0,64-crop_shape[0]-1)
    offsetY = np.rand.randint(0,64-crop_shape[1]-1)
    offsetZ = np.rand.randint(0,64-crop_shape[2]-1)
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
    msks = np.zeros_like(imgs)

    classes = hdf5_file["output"][random_idx, 0]

    return imgs, msks, classes

def generate_data(hdf5_file, batch_size=50, subset=0, validation=False):
    """Replaces Keras' native ImageDataGenerator."""
    """ Randomly select batch_size rows from the hdf5 file dataset """

    # If validation, then get the subset
    # If not validation (training), then get everything but the subset.
    if validation:
        idx_master = get_idx_for_onesubset(hdf5_file, subset)
    else:
        idx_master = get_idx_for_classes(hdf5_file, subset)

    input_shape = tuple([batch_size] + list(hdf5_file['input'].attrs['lshape']) + [1])
    input_shape = (batch_size, 64,64,64,1) # check this input shape for 3D also check if above can be taken out and replace with this line

    while True:

        random_idx = get_random_idx(idx_master, batch_size)
        imgs = hdf5_file["input"][random_idx,:]
        imgs = imgs.reshape(input_shape)
        imgs = np.swapaxes(imgs, 1, 3)

        if not validation:  # Training need augmentation. Validation does not.
            ## Need to augment
            imgs, msks = augment_data(imgs, msks)

        classes = hdf5_file["output"][random_idx, 0]

        yield imgs, msks  #, classes


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
    model = unet3D(crop_shape, use_upsampling=True, n_out=1, dropout=0.2,
    			print_summary = False)

    tb_log = keras.callbacks.TensorBoard(log_dir=TB_LOG_DIR,
                                histogram_freq=0,
                                batch_size=batch_size,
                                write_graph=True,
                                write_grads=True,
                                write_images=True,
                                embeddings_freq=0,
                                embeddings_layer_names=None,
                                embeddings_metadata=None)


    checkpointer = keras.callbacks.ModelCheckpoint(filepath=CHECKPOINT_FILENAME,
                                                   monitor=dice_coef_loss,
                                                   verbose=1,
                                                   save_best_only=True)

    # model.compile(optimizer='adam',
    #               loss='binary_crossentropy',
    #               metrics=['accuracy'])

    model.compile(optimizer='adam',
                  loss=dice_coef_loss,
                  metrics=['accuracy', dice_coef])

    print(model.summary())

    validation_batch_size = 64
    train_generator = generate_data(hdf5_file, batch_size, subset=HOLDOUT_SUBSET, validation=False)
    validation_generator = generate_data(hdf5_file, validation_batch_size, subset=HOLDOUT_SUBSET, validation=True)

    history = model.fit_generator(train_generator,
                        steps_per_epoch=num_rows//batch_size, epochs=2,
                        validation_data = validation_generator,
                        validation_steps = 1000,
                        callbacks=[tb_log, checkpointer])
