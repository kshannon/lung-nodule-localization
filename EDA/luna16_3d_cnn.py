import os
os.environ["CUDA_VISIBLE_DEVICES"]="1"  # Only use gpu #1 (0-4)

import tensorflow as tf

import keras
config = tf.ConfigProto()
config.gpu_options.allow_growth=True # Don't use all GPU memory if not needed
sess = tf.Session(config=config)
keras.backend.set_session(sess)

import numpy as np
import h5py
import os

# root_dir = !pwd
# s3bucket_path = root_dir[0] + '/../s3bucket_goofys/' # remote S3 via goofys
s3bucket_path = '/nfs/site/home/ganthony/'
path_to_hdf5 = s3bucket_path + '64x64x3-patch.hdf5'
hdf5_file = h5py.File(path_to_hdf5, 'r') # open in read-only mode

print("Valid hdf5 file in 'read' mode: " + str(hdf5_file))
file_size = os.path.getsize(path_to_hdf5)
print('Size of hdf5 file: {:.3f} GB'.format(file_size/2.0**30))

num_rows = hdf5_file['input'].shape[0]
print("There are {} images in the dataset.".format(num_rows))

print("The datasets within the HDF5 file are:\n {}".format(list(hdf5_file.values())))

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

def get_random_idx(hdf5_file, idx, batch_size = 20):
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

def img_rotate(img):
    '''
    Perform a random rotation on the tensor
    `img` is the tensor
    '''
    shape = img.shape
    # This will flip along n-1 axes. (If we flipped all n axes then we'd get the same result every time)
    ax = np.random.choice(len(shape)-1,2, replace=False) # Choose randomly which axes to flip
    return np.flip(img.swapaxes(ax[0], ax[1]), ax[0]) # Random +90 or -90 rotation

def img_flip(img):
    '''
    Performs a random flip on the tensor.
    If the tensor is C x H x W x D this will perform flips on two of the C, H, D dimensions
    If the tensor is C x H x W this will perform flip on either the H or the W dimension.
    `img` is the tensor
    '''
    shape = img.shape
    # This will flip along n-1 axes. (If we flipped all n axes then we'd get the same result every time)
    ax = np.random.choice(len(shape)-1,len(shape)-2, replace=False) + 1 # Choose randomly which axes to flip
    for i in ax:
        img = np.flip(img, i) # Randomly flip along all but one axis
    return img

def augment_data(imgs):
    '''
    Performs random flips, rotations, and other operations on the image tensors.
    '''

    imgs_length = imgs.shape[0]

    for idx in range(imgs_length):
        img = imgs[idx, :]

        if (np.random.rand() > 0.5):
            img = img_flip(img)

#         if (np.random.rand() > 0.5):

#             if (np.random.rand() > 0.5):
#                 img = img_rotate(img)

#             if (np.random.rand() > 0.5):
#                 img = img_flip(img)

#         else:

#             if (np.random.rand() > 0.5):
#                 img = img_flip(img)

#             if (np.random.rand() > 0.5):
#                 img = img_rotate(img)

        imgs[idx,:] = img

    return imgs

def get_batch(hdf5_file, batch_size=50, exclude_subset=0):
    """Replaces Keras' native ImageDataGenerator."""
    """ Randomly select batch_size rows from the hdf5 file dataset """

    #input_shape = tuple([batch_size] + list(hdf5_file['input'].attrs['lshape']) + [1])
    input_shape = (batch_size, 3,64,64,1)

    idx_master = get_idx_for_classes(hdf5_file, exclude_subset)

    random_idx = get_random_idx(hdf5_file, idx_master, batch_size)
    imgs = hdf5_file["input"][random_idx,:]

    imgs = imgs.reshape(input_shape)
    imgs = np.swapaxes(imgs, 1,3)
    ## Need to augment
    #imgs = augment_data(imgs)

    classes = hdf5_file["output"][random_idx, 0]

    return imgs, classes

def generate_data(hdf5_file, batch_size=50, exclude_subset=0):
    """Replaces Keras' native ImageDataGenerator."""
    """ Randomly select batch_size rows from the hdf5 file dataset """

    input_shape = tuple([batch_size] + list(hdf5_file['input'].attrs['lshape']) + [1])
    input_shape = (batch_size, 3,64,64,1)

    idx_master = get_idx_for_classes(hdf5_file, exclude_subset)

    while True:

        random_idx = get_random_idx(hdf5_file, idx_master, batch_size)
        imgs = hdf5_file["input"][random_idx,:]
        imgs = imgs.reshape(input_shape)
        imgs = np.swapaxes(imgs, 1,3)
        ## Need to augment
        imgs = augment_data(imgs)

        classes = hdf5_file["output"][random_idx, 0]

        yield imgs, classes

input_shape = tuple(list(hdf5_file["input"].attrs["lshape"]))
batch_size = 512   # Batch size to use
print (input_shape)

from resnet3d import Resnet3DBuilder

model = Resnet3DBuilder.build_resnet_18((64, 64, 3, 1), 1)  # (input tensor shape, number of outputs)

tb_log = keras.callbacks.TensorBoard(log_dir='./tb_3D_logs', histogram_freq=0, batch_size=batch_size,
                            write_graph=True,
                            write_grads=True, write_images=True,
                            embeddings_freq=0, embeddings_layer_names=None,
                            embeddings_metadata=None)

import time
CHECKPOINT_FILENAME = "cnn_3d_64_64_3" + time.strftime("_%Y%m%d_%H%M%S") + "hdf5"
checkpointer = keras.callbacks.ModelCheckpoint(filepath=CHECKPOINT_FILENAME, verbose=1, save_best_only=True)

model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

print(model.summary())

history = model.fit_generator(generate_data(hdf5_file, batch_size, exclude_subset=2),
                    steps_per_epoch=num_rows//batch_size, epochs=6,
                    callbacks=[tb_log, checkpointer])
