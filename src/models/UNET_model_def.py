import keras
data_format = "channels_last"
concat_axis = -1

def unet3D_modelB(input_img, use_upsampling=True, n_out=1, dropout=0.2,
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
	conv3 = keras.layers.Dropout(dropout)(conv3) ### Trying dropout layers earlier on, as indicated in the paper
	conv3 = keras.layers.Conv3D(name="conv3b", filters=256, **params)(conv3)
	conv3 = keras.layers.BatchNormalization(axis =-1)(conv3)
	conv3 = keras.layers.Activation('relu')(conv3)
	pool3 = keras.layers.MaxPooling3D(name="pool3", pool_size=(2, 2, 2))(conv3)

	conv4 = keras.layers.Conv3D(name="conv4a", filters=256, **params)(pool3)
	conv4 = keras.layers.BatchNormalization(axis =-1)(conv4)
	conv4 = keras.layers.Activation('relu')(conv4)
	conv4 = keras.layers.Dropout(dropout)(conv4) ### Trying dropout layers earlier on, as indicated in the paper
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

	conv7 = keras.layers.Conv3D(name="conv7a", filters=64, **params)(up6)
	conv7 = keras.layers.BatchNormalization(axis =-1)(conv7)
	conv7 = keras.layers.Activation('relu')(conv7)
	conv7 = keras.layers.Conv3D(name="conv7b", filters=32, **params)(conv7)
	conv7 = keras.layers.BatchNormalization(axis =-1)(conv7)
	conv7 = keras.layers.Activation('relu')(conv7)

	pred_msk = keras.layers.Conv3D(name="PredictionMask", filters=n_out, kernel_size=(1, 1, 1),
					data_format=data_format, activation="sigmoid")(conv7)

	#Branch is created from conv7 which are feature maps
	#But global avg pooling on feature maps is not helping and hence changing back to pred_msk
	class_pred = keras.layers.GlobalAveragePooling3D(name='PredictionClass')(pred_msk)

	model = keras.models.Model(inputs=[inputs], outputs=[pred_msk,class_pred])

	if print_summary:
		#model = keras.models.Model(inputs=[inputs], outputs=[class_pred])
		model.summary()

	# return pred
	return model

def unet3D_modelC(input_img, use_upsampling=True, n_out=1, dropout=0.2,
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
	conv3 = keras.layers.Dropout(dropout)(conv3) ### Trying dropout layers earlier on, as indicated in the paper
	conv3 = keras.layers.Conv3D(name="conv3b", filters=256, **params)(conv3)
	conv3 = keras.layers.BatchNormalization(axis =-1)(conv3)
	conv3 = keras.layers.Activation('relu')(conv3)
	pool3 = keras.layers.MaxPooling3D(name="pool3", pool_size=(2, 2, 2))(conv3)

	conv4 = keras.layers.Conv3D(name="conv4a", filters=256, **params)(pool3)
	conv4 = keras.layers.BatchNormalization(axis =-1)(conv4)
	conv4 = keras.layers.Activation('relu')(conv4)
	conv4 = keras.layers.Dropout(dropout)(conv4) ### Trying dropout layers earlier on, as indicated in the paper
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

	conv7 = keras.layers.Conv3D(name="conv7a", filters=64, **params)(up6)
	conv7 = keras.layers.BatchNormalization(axis =-1)(conv7)
	conv7 = keras.layers.Activation('relu')(conv7)
	conv7 = keras.layers.Conv3D(name="conv7b", filters=32, **params)(conv7)
	conv7 = keras.layers.BatchNormalization(axis =-1)(conv7)
	conv7 = keras.layers.Activation('relu')(conv7)

	pred_msk = keras.layers.Conv3D(name="PredictionMask", filters=n_out, kernel_size=(1, 1, 1),
					data_format=data_format, activation="sigmoid")(conv7)

	#Branch is created from conv7 which are feature maps
	#But global avg pooling on feature maps is not helping and hence changing back to pred_msk
	class_pred = keras.layers.GlobalAveragePooling3D(name='PredictionClass')(pred_msk)

	model = keras.models.Model(inputs=[inputs], outputs=[pred_msk,class_pred])

	if print_summary:
		#model = keras.models.Model(inputs=[inputs], outputs=[class_pred])
		model.summary()

	# return pred
	return model


def unet3D_ModelB_exp1(input_img, use_upsampling=True, n_out=1, dropout=0.2,
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
	conv3 = keras.layers.Dropout(dropout)(conv3) ### Trying dropout layers earlier on, as indicated in the paper
	conv3 = keras.layers.Conv3D(name="conv3b", filters=256, **params)(conv3)
	conv3 = keras.layers.BatchNormalization(axis =-1)(conv3)
	conv3 = keras.layers.Activation('relu')(conv3)
	pool3 = keras.layers.MaxPooling3D(name="pool3", pool_size=(2, 2, 2))(conv3)

	conv4 = keras.layers.Conv3D(name="conv4a", filters=256, **params)(pool3)
	conv4 = keras.layers.BatchNormalization(axis =-1)(conv4)
	conv4 = keras.layers.Activation('relu')(conv4)
	conv4 = keras.layers.Dropout(dropout)(conv4) ### Trying dropout layers earlier on, as indicated in the paper
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

	conv7 = keras.layers.Conv3D(name="conv7a", filters=64, **params)(up6)
	conv7 = keras.layers.BatchNormalization(axis =-1)(conv7)
	conv7 = keras.layers.Activation('relu')(conv7)
	conv7 = keras.layers.Conv3D(name="conv7b", filters=32, **params)(conv7)
	conv7 = keras.layers.BatchNormalization(axis =-1)(conv7)
	conv7 = keras.layers.Activation('relu')(conv7)

	pred_msk = keras.layers.Conv3D(name="PredictionMask", filters=n_out, kernel_size=(1, 1, 1),
					data_format=data_format, activation="sigmoid")(conv7)

	#Branch is created from conv7 which are feature maps
	#But global avg pooling on feature maps is not helping and hence changing back to pred_msk
	# class_pred = keras.layers.GlobalAveragePooling3D(name='PredictionClass')(pred_msk)

	model = keras.models.Model(inputs=[inputs], outputs=[pred_msk])
	# model = keras.models.Model(inputs=[inputs], outputs=[pred_msk,class_pred])

	if print_summary:
		#model = keras.models.Model(inputs=[inputs], outputs=[class_pred])
		model.summary()

	# return pred
	return model

def unet3D_sizeagnostic_Model16(use_upsampling=True, n_out=1, dropout=0.2,
			print_summary = False):
	"""
	3D U-Net model
	"""
	print("3D U-Net Segmentation")
	# Set keras learning phase to train
	keras.backend.set_learning_phase(True)

	# Don"t initialize variables on the fly
	keras.backend.manual_variable_initialization(False)

	input_img = (None, None, None,1)
	inputs = keras.layers.Input(shape=input_img, name="Input_Image")

	# Use below if wanted to use batch normalization and Relu activation separately
	params = dict(kernel_size=(3, 3, 3), activation=None,
				  padding="same", data_format=data_format,
				  kernel_initializer="he_uniform")

	# params = dict(kernel_size=(3, 3, 3), activation="relu",
	# 			  padding="same", data_format=data_format,
	# 			  kernel_initializer="he_uniform")

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
	conv3 = keras.layers.Dropout(dropout)(conv3) ### Trying dropout layers earlier on, as indicated in the paper
	conv3 = keras.layers.Conv3D(name="conv3b", filters=256, **params)(conv3)
	conv3 = keras.layers.BatchNormalization(axis =-1)(conv3)
	conv3 = keras.layers.Activation('relu')(conv3)
	pool3 = keras.layers.MaxPooling3D(name="pool3", pool_size=(2, 2, 2))(conv3)

	conv4 = keras.layers.Conv3D(name="conv4a", filters=256, **params)(pool3)
	conv4 = keras.layers.BatchNormalization(axis =-1)(conv4)
	conv4 = keras.layers.Activation('relu')(conv4)
	conv4 = keras.layers.Dropout(dropout)(conv4) ### Trying dropout layers earlier on, as indicated in the paper
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

	conv7 = keras.layers.Conv3D(name="conv7a", filters=64, **params)(up6)
	conv7 = keras.layers.BatchNormalization(axis =-1)(conv7)
	conv7 = keras.layers.Activation('relu')(conv7)
	conv7 = keras.layers.Conv3D(name="conv7b", filters=32, **params)(conv7)
	conv7 = keras.layers.BatchNormalization(axis =-1)(conv7)
	conv7 = keras.layers.Activation('relu')(conv7)

	pred_msk = keras.layers.Conv3D(name="PredictionMask", filters=n_out, kernel_size=(1, 1, 1),
					data_format=data_format, activation="sigmoid")(conv7)

	#Branch is created from conv7 which are feature maps
	#But global avg pooling on feature maps is not helping and hence changing back to pred_msk
	# class_pred = keras.layers.GlobalAveragePooling3D(name='PredictionClass')(pred_msk)

	model = keras.models.Model(inputs=[inputs], outputs=[pred_msk, conv7])
	# model = keras.models.Model(inputs=[inputs], outputs=[pred_msk,class_pred])

	if print_summary:
		#model = keras.models.Model(inputs=[inputs], outputs=[class_pred])
		model.summary()

	# return pred
	return model

def unet3D_Model6_Model13(input_img, use_upsampling=True, n_out=1, dropout=0.2,
			print_summary = False):
	"""
	3D U-Net model
	"""
	print("3D U-Net Segmentation")
	# Set keras learning phase to train
	keras.backend.set_learning_phase(True)

	# Don"t initialize variables on the fly
	keras.backend.manual_variable_initialization(False)

	# Missed to add below line
	# input_img = (None, None, None, 1)
	inputs = keras.layers.Input(shape=input_img, name="Input_Image")

	# Use below if wanted to use batch normalization and Relu activation separately
	params = dict(kernel_size=(3, 3, 3), activation=None,
				  padding="same", data_format=data_format,
				  kernel_initializer="he_uniform")

	# params = dict(kernel_size=(3, 3, 3), activation="relu",
	# 			  padding="same", data_format=data_format,
	# 			  kernel_initializer="he_uniform")

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
	conv3 = keras.layers.Dropout(dropout)(conv3) ### Trying dropout layers earlier on, as indicated in the paper
	conv3 = keras.layers.Conv3D(name="conv3b", filters=256, **params)(conv3)
	conv3 = keras.layers.BatchNormalization(axis =-1)(conv3)
	conv3 = keras.layers.Activation('relu')(conv3)
	pool3 = keras.layers.MaxPooling3D(name="pool3", pool_size=(2, 2, 2))(conv3)

	conv4 = keras.layers.Conv3D(name="conv4a", filters=256, **params)(pool3)
	conv4 = keras.layers.BatchNormalization(axis =-1)(conv4)
	conv4 = keras.layers.Activation('relu')(conv4)
	conv4 = keras.layers.Dropout(dropout)(conv4) ### Trying dropout layers earlier on, as indicated in the paper
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

	conv7 = keras.layers.Conv3D(name="conv7a", filters=64, **params)(up6)
	conv7 = keras.layers.BatchNormalization(axis =-1)(conv7)
	conv7 = keras.layers.Activation('relu')(conv7)
	conv7 = keras.layers.Conv3D(name="conv7b", filters=32, **params)(conv7)
	conv7 = keras.layers.BatchNormalization(axis =-1)(conv7)
	conv7 = keras.layers.Activation('relu')(conv7)

	pred_msk = keras.layers.Conv3D(name="PredictionMask", filters=n_out, kernel_size=(1, 1, 1),
					data_format=data_format, activation="sigmoid")(conv7)

	#Branch is created from conv7 which are feature maps
	#But global avg pooling on feature maps is not helping and hence changing back to pred_msk
	# class_pred = keras.layers.GlobalAveragePooling3D(name='PredictionClass')(pred_msk)

	model = keras.models.Model(inputs=[inputs], outputs=[pred_msk, conv7])
	# model = keras.models.Model(inputs=[inputs], outputs=[pred_msk,class_pred])

	if print_summary:
		#model = keras.models.Model(inputs=[inputs], outputs=[class_pred])
		model.summary()

	# return pred
	return model
