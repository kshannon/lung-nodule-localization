
# coding: utf-8

# #### To test: 
# 1. Create a folder ../data/luna16/
# 2. Create a folder ../data/luna16/subset2
#     -Under this folder copy one scan for testing (script will process all the scan at this location) 
#       1.3.6.1.4.1.14519.5.2.1.6279.6001.100621383016233746780170740405.mhd & raw file 
#       (Google drive https://drive.google.com/drive/u/1/folders/13wmubTgm-7sh3MxPGxqmVZuoqi0G3ufW
# 3. Create a folder ../data/luna16/hdf5
#     -Under this copy UNET_weights_H5.h5 (download from google drive)

# In[2]:

import pandas as pd
import numpy as np
import h5py
import pandas as pd
import argparse
import SimpleITK as sitk
from PIL import Image
import os, glob 
import os, os.path
import tensorflow as tf
import keras

from ipywidgets import interact
import json
import pickle
from datetime import datetime
from tqdm import tqdm, trange

from UNET_utils import *
get_ipython().magic(u'matplotlib inline')


# In[3]:

# import argparse
# parser = argparse.ArgumentParser(description='Prediction on HOLDOUT subset',add_help=True)
# parser.add_argument("--holdout", type=int, default=0, help="HOLDOUT subset for predictions")
# args = parser.parse_args()
# HOLDOUT = args.holdout


# In[4]:

HOLDOUT = 5
HO_dir = 'HO{}/'.format(HOLDOUT)
data_dir = '../data/luna16/'
model_wghts = 'hdf5/UNET_weights_H{}.h5'.format(HOLDOUT)


# In[5]:

PADDED_SIZE = (448, 448, 368)
SLICES = 8
TILE_SIZE = (448,448,SLICES)


# In[6]:

def model_create_loadWghts_Model_A(img_size=TILE_SIZE):
    input_shape = tuple(list(img_size) + [1])
    model = create_unet3D_Model_A(input_shape, use_upsampling=True)

    model.load_weights(data_dir + model_wghts)
    model.compile(optimizer='adam',
                  loss=[dice_coef_loss],
                  metrics= [dice_coef])
    return model


# In[7]:

def model_create_loadWghts(img_size=TILE_SIZE):
    input_shape = tuple(list(img_size) + [1])
    model = create_UNET3D(input_shape, use_upsampling=True)

    model.load_weights(data_dir + model_wghts)
#   ##Uncomment the followng line when just want to Transfer Weights to matching layers
#     model.load_weights(data_dir + model_wghts, by_name=True)  
    model.compile(optimizer='adam',
                  loss={'PredictionMask': dice_coef_loss, \
                        'PredictionClass': 'binary_crossentropy'}, \
                  loss_weights={'PredictionMask': 0.8, 'PredictionClass': 0.2},
                  metrics={'PredictionMask':dice_coef,'PredictionClass': 'accuracy'})

    return model


# In[8]:

def find_mask(model, padded_img):
    print ()
    predicted_mask = np.zeros(PADDED_SIZE)
    print ("Total tiles : {}".format(PADDED_SIZE[2]//SLICES))

    for i in  tqdm(range( PADDED_SIZE[2]//SLICES), total=PADDED_SIZE[2]//SLICES, unit="tiles"):
#         print ("Processing tile number : {}".format(i))
        tile = padded_img[:, :, (i*SLICES) : SLICES*(i+1)]
        tile = tile.reshape(tuple([1] + list (tile.shape) + [1]))
        tile_predictions = model.predict(tile, verbose=2)
        
        tile_mask = tile_predictions[0].reshape(TILE_SIZE)
        predicted_mask[:, :, (i*SLICES) : SLICES*(i+1)] = tile_mask
    return predicted_mask


# In[9]:

get_ipython().run_cell_magic(u'time', u'', u't0 = datetime.now()\npredictions_dict = {}\nsize_dict = {}\nmodel = model_create_loadWghts_Model_A(TILE_SIZE) \nfileCount = len(glob.glob(data_dir + \'subset2/\' + \'*.mhd\'))\n                \nfor f in tqdm(glob.glob(data_dir + \'subset2/\' + \'*.mhd\'), total=fileCount, unit="files") :\n    print ("\\n Processing scan file: {}".format(os.path.basename(f)))\n    seriesuid = os.path.splitext(os.path.basename(f))[0]\n    # Step-1\n    itk_img = sitk.ReadImage(f) \n    img_np_array = sitk.GetArrayFromImage(itk_img)\n    original_size = img_np_array.shape\n    print ("Original-Size of loaded image : {}".format(original_size))\n    # Step-2 \n    itk_img_norm = normalize_img(itk_img)\n    img_np_array_norm = sitk.GetArrayFromImage(itk_img_norm)\n    normalized_size = img_np_array_norm.shape\n    # Step-3 \n    img = img_np_array_norm.copy()\n#     img = normalize_HU(img_np_array_norm)\n    img = np.swapaxes(img, 0,2)   ##needed as SITK swaps axis  \n    print ("Normalized input image size: {}".format(img.shape))\n    # Step-4   # Step-5\n    padded_img = np.zeros(PADDED_SIZE)\n    padded_img[ :img.shape[0], :img.shape[1], :img.shape[2] ] = img\n    print ("Padded-image size: {}".format(padded_img.shape))\n    \n    predicted_mask = find_mask(model, padded_img)\n    predictions_dict[seriesuid] = (img.shape, padded_img, predicted_mask)\n    size_dict[seriesuid] = img.shape\n\nprint(\'Predicted Mask sum for entire scan: {}\'.format(np.sum(predicted_mask)))\npickle.dump(predictions_dict, open(\'Model_A_noHU_entire_predictions_{}.dat\'.format(seriesuid), \'wb\'))\npickle.dump(size_dict, open(\'Model_A_noHU_entire_size_{}.dat\'.format(seriesuid), \'wb\'))    \nprint(\'Processing runtime: {}\'.format(datetime.now() - t0))')


# In[12]:

# def displaySlice(sliceNo):
    
#     plt.figure(figsize=[20,20]);    
#     plt.subplot(121)
#     plt.title("True Image")
#     plt.imshow(padded_img[:, :, sliceNo], cmap='bone');

#     plt.subplot(122)
#     plt.title("Predicted Mask")
#     plt.imshow(predicted_mask[:, :, sliceNo], cmap='bone');
#     plt.show()
# interact(displaySlice, sliceNo=(1,img.shape[2],1));


# In[ ]:




# ###### Following sections for reference & WIP code snippets -AL

# In[ ]:

## Multiple tile test....performance hog, so exploiting the GPU for entire slice without compromising predictions 
##and for better performance  -AL

# slices = 16
# predicted_img = np.zeros(padded_size)

# for i in range(368//slices):
#     tile_1 = padded_img[:224, :224, (i*slices) : slices*(i+1)]
#     tile_2 = padded_img[224:, 224:, (i*slices) : slices*(i+1) ] 


# In[ ]:

# slices = 8
# predicted_mask = np.zeros(PADDED_SIZE)

# for i in range(24//SLICES):
#     tile = padded_img[:, :, (i*SLICES) : SLICES*(i+1)]
#     tile = tile.reshape(tuple([1] + list (tile.shape) + [1]))
# #     print(tile.shape)

#     tile_predictions = model.predict(tile, verbose=2)
#     tile_mask = tile_predictions[0].reshape(448, 448, 8)
    
#     print (tile_mask.shape)
#     predicted_mask[:, :, (i*SLICES) : SLICES*(i+1)] = tile_mask


# In[ ]:

# slices = 8
# test_slice = padded_img[:, :, :slices]
# print(test_slice.shape)
# model = model_create_loadWghts(test_slice.shape) 
# # slice_predictions = model.predict(test_slice, verbose=2)


# In[ ]:

# print ("Shape of predicted mask or segmented image : {}".format(predictions_small_img[0].shape))
# print ("Shape of predicted class : {}".format(predictions_small_img[1].shape))
# predictions_small_img[0] [:, 25 : 26, :]


# In[ ]:

# ## AL - TEST : making an image of size 48,48,48 with random 0 or 1
# ### Case 2 : As a test created an input image of size (1, 48,48,48,1) 
# # with random 0 or 1; this works fine and able to create predictions successfully
# t2 =  np.random.choice(2,(48,48,48))
# t2 = t2.reshape(tuple([1] + list (t2.shape) + [1]))

# print ("Shape of test input image : {}".format(t2.shape))
# predictions = model.predict(t2, verbose=2)

# print ("Shape of predicted mask or segmented image : {}".format(predictions[0].shape))
# print ("Shape of predicted class : {}".format(predictions[1].shape))
# # predictions[0] [:, 25 : 26, :]


# In[ ]:

# padded_img[225:232, 225:232, 175]
# predicted_mask[225:232, 225:232, 175]

