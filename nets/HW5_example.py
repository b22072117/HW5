from __future__ import print_function
import tensorflow as tf
import numpy as np
import scipy.ndimage
from scipy import misc

import sys
import os
import time
import pdb

###########################
##***********************##
## Author : Hong-Yi Chen ##
## Date   : 2017.12.02   ##
##***********************##
###########################

#========================#
#    Global Parameter    #
#========================#
IS_TRAINING = True
IS_TESTING = True
BATCH_SIZE = 4
CLASS_NUM  = 12
EPOCH = 50
Dataset_Path = '../dataset/CamVid'


#===========#
#    Def    #
#===========#
def CamVid_dataset_parser(
	data_index,
	target_index
	):
	
	"""
	Simultaneously read out the data and target.
	Also, do some image pre-processing. 
	(You can add your own image pre-processing)
	
	Args :
	(1) data_index   : The array of the data images name.
	(2) target_index : The array of the target images name.
	
	Returns:
	(1) data   : The data images. Array Shape = [Total_Image_Num, Image_Height, Image_Width, Image_Depth (3 for RGB)]
	(2) target : The target images. Array Shape = [Total_Image_Num, Image_Height, Image_Width, Image_Depth (3 for RGB)]
	"""
	
	# data_index : The name of all the images
	for iter in range(len(data_index)):
		# get each image name
		data_name   = data_index[iter]
		target_name = target_index[iter]
	
		# Read Image
		data_tmp   = misc.imread(Dataset_Path + data_name)
		target_tmp = misc.imread(Dataset_Path + target_name)
		
		# Data Preprocessing : 
		# You can add your own image preprocessing below.
		"""
		"""
		# If you want to resize the input size, you can uncomment the following code.
		# - H_resize: The height of the image you want to resize.
		# - W_resize: The width of the image you want to resize.
		
		H_resize = 224
		W_resize = 224
		data_tmp   = scipy.misc.imresize(data_tmp,   (H_resize, W_resize))
		target_tmp = scipy.misc.imresize(target_tmp, (H_resize, W_resize))
		
		
		# Concatenate each data to a big array 
		# Final shape of array : [Total_Image_Num, Image_Height, Image_Width, Image_Depth (3 for RGB)]
		if iter==0:
			data   = np.expand_dims(data_tmp  , axis=0)
			target = np.expand_dims(target_tmp, axis=0)
		else:
			data   = np.concatenate([data  , np.expand_dims(data_tmp  , axis=0)], axis=0)
			target = np.concatenate([target, np.expand_dims(target_tmp, axis=0)], axis=0)
			
	return data, target

def one_hot(
	target,
	class_num
	):
	
	"""
	Modify the value to the one-of-k type.
	
	i.e. array([[1, 2]
	            [3, 4]])
	
	args:
	(1) target    : The 4-D array of the image. Array Shape = [Image_Num, Image_Height, Image_Width, Image_Depth]
	(2) class_num : The number of the class. (i.e. 12 for the CamVid dataset; 10 for the mnist dataset)
	
	Returns:
	(1) one_hot_target: The 4-D array of the one-of-k type target. Array Shape = [Image_Num, Image_Height, Image_Width, class_num]
	"""
	
	target.astype('int64')
	one_hot_target = np.zeros([np.shape(target)[0], np.shape(target)[1], np.shape(target)[2], class_num])
	
	meshgrid_target = np.meshgrid(np.arange(np.shape(target)[1]), np.arange(np.shape(target)[0]), np.arange(np.shape(target)[2]))
	
	one_hot_target[meshgrid_target[1], meshgrid_target[0], meshgrid_target[2], target] = 1
	
	return one_hot_target
	
def compute_accuracy(
	xs,
	ys,
	is_training,
	prediction,
	v_xs,
	v_ys, 
	batch_size,
	sess
	):
	
	"""
	Compute the accuray of the semantic segmentation classification problem.
	
	Args:
	(1) xs          : A 4-D input placeholder tensor. Shape = [Batch_Size, Image_Height Image_Width, Image_Depth]
	(2) ys          : A target placeholder tensor. Shape = [Batch_Size, [output_shape]] (output_shape is user-defined)
	(3) is_training : A 1-D bool type placeholder tensor.
	(4) prediction  : An output placeholder tensor. Shape = Must be same as the ys shape.
	(5) v_xs        : An 4-D array of the data. Shape = [Image_Num, Image_Height Image_Width, Image_Depth]
	(6) v_ys        : An array of the target. Shape = [Image_Num, [target_shape]]
	(7) batch_size  : The data number in one batch.
	(8) sess        : Tensorflow Session.
	
	Returns:
	(1) total_accuracy_top1: Top1 Accuracy.
	"""
	
	total_iter = int(len(v_xs)/batch_size)
	total_accuracy_top1 = 0
	
	for iter in range(total_iter):
		v_xs_part = v_xs[iter*batch_size:(iter+1)*batch_size, :]
		v_ys_part = v_ys[iter*batch_size:(iter+1)*batch_size, :]
		
		y_pre = sess.run( prediction, 
					      feed_dict={ xs: v_xs_part, 
					      			  ys: v_ys_part,											
					      			  is_training: False})
		
		
		prediction_top1 = np.argsort(-y_pre, axis=-1)[:, :, :, 0]
		
		# Calculate the Accuracy
		correct_prediction_top1 = np.equal(prediction_top1, np.argmax(v_ys_part, -1))
		accuracy_top1 = np.mean(correct_prediction_top1.astype(float))
		total_accuracy_top1 = total_accuracy_top1 + accuracy_top1
		
		# Save the result as array
		if iter==0:
			result = prediction_top1
		else:
			result = np.concatenate([result, prediction_top1], axis=0)
		
		
	total_accuracy_top1 = total_accuracy_top1 / total_iter
	
	return result, total_accuracy_top1

# (Optinal Function)
def shuffle_data(
	data, 
	target
	):
	
	"""
	Shuffle the data.
	
	Args:
	(1) data   : An 4-D array of the data. Shape = [Image_Num, Image_Height Image_Width, Image_Depth]
	(2) target : An array of the target. Shape = [Image_Num, [target_shape]]
	Returns:
	(1) shuffle_data   : Shuffle data. Shape = [Image_Num, Image_Height Image_Width, Image_Depth]
	(2) shuffle_target : Shuffle target. Shape = [Image_Num, [target_shape]]
	"""
	
	shuffle_index = np.arange(np.shape(data)[0])
	np.random.shuffle(shuffle_index)
	
	shuffle_data   = data  [shuffle_index, :, :, :]
	shuffle_target = target[shuffle_index, :, :, :]
	
	return shuffle_data, shuffle_target

def per_class_accuracy(
	prediction, 
	batch_ys
	):
	
	"""
	Show the accuracy of each class.
	
	Args:
	(1) prediction : An output placeholder tensor. Shape = Must be same as the batch_ys shape.
	(2) batch_ys   : An array of the target. Shape = [Batch_Size, [target_shape]]
	
	Returns:
	(None)
	"""
	
	print("Per Class Accuracy")
	[BATCH, HEIGHT, WIDTH, CLASS_NUM] = np.shape(batch_ys)
	correct_num = np.zeros([CLASS_NUM, 1])
	total_num = np.zeros([CLASS_NUM, 1])
	
	print_per_row = 10
	cn = np.zeros([print_per_row], np.int32)
	tn = np.zeros([print_per_row], np.int32)

	for i in range(CLASS_NUM):
		y_tmp = np.equal(np.argmax(batch_ys, -1), i)
		p_tmp = np.equal(np.argmax(prediction, -1), i)
		total_num = np.count_nonzero(y_tmp)
		zeros_num = np.count_nonzero( (p_tmp+y_tmp) == 0)
		correct_num = np.count_nonzero(np.equal(y_tmp, p_tmp)) - zeros_num
		if total_num == 0:
			accuracy = -1
		else:
			accuracy = float(correct_num) / float(total_num)
		
		if CLASS_NUM <= 15:
			print("    Class{Iter}	: {predict} / {target}".format(Iter = i, predict=correct_num, target=total_num))
		else:
			iter = i%print_per_row
			cn[iter] = correct_num
			tn[iter] = total_num
			if i%print_per_row==0:
				print("    Class{Iter}	: {predict} / {target}".format(Iter = i, predict=np.sum(cn), target=np.sum(tn)))

def save_CamVid_result_as_image(
	result,
	path, 
	file_index
	):
	
	"""
	Save the CamVid result to the image for visualing.
	
	Args:
	(1) result     : Image to be Save. An 4D array. Shape=[Image_Num, Image_Height, Image_Width, Image_Depth]
	(2) Path       : Path to save the image.
	(3) file_index : Index of each images.
	
	Returns:
	(None)
	"""
	
	# -- Color the result --
	print("Coloring the results ... ")
	#***************************************#
	#	class0 : (	128 	128 	128	)	#
	#	class1 : (	128 	0 		0	)	#
	#	class2 : (	192 	192 	128	)	#
	#	class3 : (	128 	64 		128	)	#
	#	class4 : (	0 		0 		192	)	#
	#	class5 : (	128 	128 	0	)	#
	#	class6 : (	192 	128 	128	)	#
	#	class7 : (	64 		64 		128	)	#
	#	class8 : (	64 		0 		128	)	#
	#	class9 : (	64 		64 		0	)	#
	#	class10 : (	0		128 	192	)	#
	#	class11 : (	0		0		0	)	#
	#***************************************#
	shape = np.shape(result)
	RGB = np.zeros([shape[0], shape[1], shape[2], 3], np.uint8)
	for i in range(shape[0]):
		for x in range(shape[1]):
			for y in range(shape[2]):
				if result[i][x][y] == 0:
					RGB[i][x][y][0] = np.uint8(128)
					RGB[i][x][y][1] = np.uint8(128)
					RGB[i][x][y][2] = np.uint8(128)
				elif result[i][x][y] == 1:
					RGB[i][x][y][0] = np.uint8(128) 
					RGB[i][x][y][1] = np.uint8(0)
					RGB[i][x][y][2] = np.uint8(0) 
				elif result[i][x][y] == 2:
					RGB[i][x][y][0] = np.uint8(192)
					RGB[i][x][y][1] = np.uint8(192)
					RGB[i][x][y][2] = np.uint8(128)
				elif result[i][x][y] == 3:
					RGB[i][x][y][0] = np.uint8(128)
					RGB[i][x][y][1] = np.uint8(64)
					RGB[i][x][y][2] = np.uint8(128)
				elif result[i][x][y] == 4:
					RGB[i][x][y][0] = np.uint8(0)
					RGB[i][x][y][1] = np.uint8(0)
					RGB[i][x][y][2] = np.uint8(192)
				elif result[i][x][y] == 5:
					RGB[i][x][y][0] = np.uint8(128)
					RGB[i][x][y][1] = np.uint8(128)
					RGB[i][x][y][2] = np.uint8(0)
				elif result[i][x][y] == 6:
					RGB[i][x][y][0] = np.uint8(192)
					RGB[i][x][y][1] = np.uint8(128)
					RGB[i][x][y][2] = np.uint8(128)
				elif result[i][x][y] == 7:
					RGB[i][x][y][0] = np.uint8(64)
					RGB[i][x][y][1] = np.uint8(64)
					RGB[i][x][y][2] = np.uint8(128)
				elif result[i][x][y] == 8:
					RGB[i][x][y][0] = np.uint8(64)
					RGB[i][x][y][1] = np.uint8(0)
					RGB[i][x][y][2] = np.uint8(128)
				elif result[i][x][y] == 9:
					RGB[i][x][y][0] = np.uint8(64)
					RGB[i][x][y][1] = np.uint8(64)
					RGB[i][x][y][2] = np.uint8(0)
				elif result[i][x][y] == 10:
					RGB[i][x][y][0] = np.uint8(0)
					RGB[i][x][y][1] = np.uint8(128)
					RGB[i][x][y][2] = np.uint8(192)
				elif result[i][x][y] == 11:
					RGB[i][x][y][0] = np.uint8(0)
					RGB[i][x][y][1] = np.uint8(0)
					RGB[i][x][y][2] = np.uint8(0)
	
	# -- Save the result into image --
	# Create the directory if it is not exist
	if not os.path.exists(path):
		print("\033[1;35;40m%s\033[0m is not exist!" %path)
		os.mkdir(path)
		print("\033[1;35;40m%s\033[0m is created" %path)
		
	for i, target in enumerate(RGB):
		# Create the directory if it is not exist
		dir = file_index[i].split('/')
		dir_num = len(dir)
		for iter in range(1, dir_num-1):
			if not os.path.exists(path + '/' + dir[iter]):
				print("\033[1;35;40m%s\033[0m is not exist!" %path + '/' + dir[iter])
				os.mkdir(path + '/' + dir[iter])
				print("\033[1;35;40m%s\033[0m is created" %path + '/' + dir[iter])
		
		# save
		scipy.misc.imsave(path + file_index[i], target)

#========================#
#    Model Components    #
#========================#
def conv2D(
	net,
	kernel_size,
	strides, 
	output_channel,
	initializer = tf.contrib.layers.variance_scaling_initializer(), 
	padding = "SAME", 
	scope = None
	):
	
	"""
	2D Convolution in CNN.
	
	Args:
	(1) net            : Input. An 4D tensor. Shape=[Batch_Size, Image_Height, Image_Width, Image_Depth]
	(2) kernel_size    : Kernel Size of the Convolution. (i.e. 3)
	(3) strides        : Stride of the Convolution. (i.e. 2)
	(4) output_channel : Output Channel of the Convolution. (i.e. 64)
	(5) initializer    : Initialization Strategy of the weights and biases variables.
	(6) padding        : Convolution Padding type. ('SAME'/'VALID')
	(7) scope          : Scope name.
	
	Returns:
	(1) net : An output tensor after convolution.
	"""
	
	input_channel = net.get_shape().as_list()[-1]
	with tf.variable_scope(scope):
		# Define Weights Variable
		weights = tf.get_variable( name        = "weights", 
		                           shape       = [kernel_size, kernel_size, input_channel, output_channel],
								   dtype       = tf.float32,
								   initializer = initializer)
		# Define Biases Variable
		#biases = tf.Variable( initial_value = tf.constant(value = 0.0, shape = [output_channel], dtype = tf.float32), 
		#                      trainable     = True, 
		#					  name           = 'biases')				
		biases = tf.get_variable( name        = "biases", 
		                          shape       = [output_channel],
								  dtype       = tf.float32,
								  initializer = initializer)
		
		tf.add_to_collection("weights", weights)
		
		# Convolution  
		net = tf.nn.conv2d( input = net, 
		                    filter = weights,
							strides = [1, strides, strides, 1],
							padding = padding,
							name    = 'conv')
		# Add Biases
		net = tf.nn.bias_add(net, biases)
		
	return net
	
def max_pooling(
	net,
	kernel_size,
	strides,
	padding = "SAME"
	):
	
	"""
	Max polling with indices.
	
	Args:
	(1) net         : Input. An 4D tensor. Shape=[Batch_Size, Image_Height, Image_Width, Image_Depth]
    (2) kernel_size : Kernel Size of the Convolution. (i.e. 2)
    (3) strides     : Stride of the Convolution. (i.e. 2)
    (4) padding     : Pooling Padding type. ('SAME'/'VALID')
	
	Return:
	(1) net         : An output tensor after max pooling.
	(2) indices     : An indices of the input net. You can see the below web-site for more detail introduction.
	(3) input_shape : The shape before pooling. It is used for the unpooling afterward.
	"""
	
	input_shape = net.get_shape().as_list()
	
	# Detail : https://www.tensorflow.org/api_docs/python/tf/nn/max_pool_with_argmax
	net, indices = tf.nn.max_pool_with_argmax( input   = net,
		                                       ksize   = [1, kernel_size, kernel_size, 1],
							                   strides = [1, strides, strides, 1],
							                   padding = padding,
							                   name    = 'max_pool')
	return net, indices, input_shape

def max_unpooling(
	net,
	output_shape,
	indices
	):
	
	"""
	Max unpolling by indices and output_shape.
	
	Args:
	(1) net          : Input. An 4D tensor. Shape=[Batch_Size, Image_Height, Image_Width, Image_Depth]
    (2) output_shape : The shape to be restored to.
    (3) indices      : An indices of the max pooling. More detail in "max_pooling"
	
	Return:
	(1) net : An output tensor after max unpooling.
	"""
	
	input_shape = net.get_shape().as_list()
	
	# Calculate indices for batch, height, width and channel
	meshgrid = tf.meshgrid(tf.range(input_shape[1]), tf.range(input_shape[0]), tf.range(input_shape[2]), tf.range(input_shape[3]))
	b = tf.cast(meshgrid[1], tf.int64)
	h = indices // (output_shape[2] * output_shape[3])
	w = indices // output_shape[3] - h * output_shape[2]
	c = indices - (h * output_shape[2] + w) * output_shape[3]
	
	# transpose indices & reshape update values to one dimension
	updates_size = tf.size(net)
	indices = tf.transpose(tf.reshape(tf.stack([b, h, w, c]), [4, updates_size]))
	values = tf.reshape(net, [updates_size])
	net = tf.scatter_nd(indices, values, output_shape)
		
	return net

def batch_norm(
	net,
	is_training
	):
	
	"""
	Batch Normalization.
	
	Args:
	(1) net         : Input. An 4D tensor. Shape=[Batch_Size, Image_Height, Image_Width, Image_Depth]
    (2) is_training : A 1-D bool type placeholder tensor.
	
	Return:
	(1) net : An output tensor after max unpooling.
	"""
	
	# Detail : https://www.tensorflow.org/api_docs/python/tf/contrib/layers/batch_norm
	net = tf.contrib.layers.batch_norm(
	   			 inputs 				= net, 
	   			 decay					= 0.95,
	   			 center					= True,
	   			 scale					= False,
	   			 epsilon				= 0.001,
	   			 activation_fn			= None,
	   			 param_initializers		= None,
	   			 param_regularizers		= None,
	   			 updates_collections	= tf.GraphKeys.UPDATE_OPS,
	   			 is_training			= True,
	   			 reuse					= None,
	   			 variables_collections	= None,
	   			 outputs_collections	= None,
	   			 trainable				= True,
	   			 batch_weights			= None,
	   			 fused					= False,
	   			 #data_format			= DATA_FORMAT_NHWC,
	   			 zero_debias_moving_mean= False,
	   			 scope					= "batch_norm",
	   			 renorm					= False,
	   			 renorm_clipping		= None,
	   			 renorm_decay			= 0.99)
	
	return net

#============#
#    Model   #
#============#
def SegNet(
	net, 
	is_training,
	initializer,
	class_num,
	scope=None
	):
	with tf.variable_scope(scope):
		with tf.variable_scope("layer0"):
			net = conv2D( net,
						  kernel_size    = 3,
						  strides        = 1, 
						  output_channel = 64,
						  initializer    = initializer, 
						  padding        = "SAME", 
						  scope          = "conv3x3")
						  
			net = batch_norm(net, is_training = is_training)
							  
			net = tf.nn.relu(features = net, name = "ReLU")		
							 
		with tf.variable_scope("layer1"): # With Pooling
			net = conv2D( net,
						  kernel_size    = 3,
						  strides        = 1, 
						  output_channel = 64,
						  initializer    = initializer, 
						  padding        = "SAME", 
						  scope          = "conv3x3")
						  
			net = batch_norm(net, is_training = is_training)
							  
			net = tf.nn.relu(features = net, name = "ReLU")

			net, indices0, unpool_shape0 = max_pooling(net, kernel_size = 2, strides = 2, padding = "SAME")
														
		with tf.variable_scope("layer2"):
			net = conv2D( net,
						  kernel_size    = 3,
						  strides        = 1, 
						  output_channel = 128,
						  initializer    = initializer, 
						  padding        = "SAME", 
						  scope          = "conv3x3")
						  
			net = batch_norm(net, is_training = is_training)
							  
			net = tf.nn.relu(features = net, name = "ReLU")
		
		with tf.variable_scope("layer3"): # With Pooling
			net = conv2D( net,
						  kernel_size    = 3,
						  strides        = 1, 
						  output_channel = 128,
						  initializer    = initializer, 
						  padding        = "SAME", 
						  scope          = "conv3x3")
						  
			net = batch_norm(net, is_training = is_training)
							  
			net = tf.nn.relu(features = net, name = "ReLU")

			net, indices1, unpool_shape1 = max_pooling(net, kernel_size = 2, strides = 2, padding = "SAME")

		with tf.variable_scope("layer4"):
			net = conv2D( net,
						  kernel_size    = 3,
						  strides        = 1, 
						  output_channel = 256,
						  initializer    = initializer, 
						  padding        = "SAME", 
						  scope          = "conv3x3")
						  
			net = batch_norm(net, is_training = is_training)
							  
			net = tf.nn.relu(features = net, name = "ReLU")

		with tf.variable_scope("layer5"):
			net = conv2D( net,
						  kernel_size    = 3,
						  strides        = 1, 
						  output_channel = 256,
						  initializer    = initializer, 
						  padding        = "SAME", 
						  scope          = "conv3x3")
						  
			net = batch_norm(net, is_training = is_training)
							  
			net = tf.nn.relu(features = net, name = "ReLU")

		with tf.variable_scope("layer6"): # With Pooling
			net = conv2D( net,
						  kernel_size    = 3,
						  strides        = 1, 
						  output_channel = 256,
						  initializer    = initializer, 
						  padding        = "SAME", 
						  scope          = "conv3x3")
						  
			net = batch_norm(net, is_training = is_training)
							  
			net = tf.nn.relu(features = net, name = "ReLU")

			net, indices2, unpool_shape2 = max_pooling(net, kernel_size = 2, strides = 2, padding = "SAME")
		
		with tf.variable_scope("layer7"):
			net = conv2D( net,
						  kernel_size    = 3,
						  strides        = 1, 
						  output_channel = 512,
						  initializer    = initializer, 
						  padding        = "SAME", 
						  scope          = "conv3x3")
						  
			net = batch_norm(net, is_training = is_training)
							  
			net = tf.nn.relu(features = net, name = "ReLU")

		with tf.variable_scope("layer8"):
			net = conv2D( net,
						  kernel_size    = 3,
						  strides        = 1, 
						  output_channel = 512,
						  initializer    = initializer, 
						  padding        = "SAME", 
						  scope          = "conv3x3")
						  
			net = batch_norm(net, is_training = is_training)
							  
			net = tf.nn.relu(features = net, name = "ReLU")

		with tf.variable_scope("layer9"): # With Pooling
			net = conv2D( net,
						  kernel_size    = 3,
						  strides        = 1, 
						  output_channel = 512,
						  initializer    = initializer, 
						  padding        = "SAME", 
						  scope          = "conv3x3")
						  
			net = batch_norm(net, is_training = is_training)
							  
			net = tf.nn.relu(features = net, name = "ReLU")

			net, indices3, unpool_shape3 = max_pooling(net, kernel_size = 2, strides = 2, padding = "SAME")

		with tf.variable_scope("layer10"):
			net = conv2D( net,
						  kernel_size    = 3,
						  strides        = 1, 
						  output_channel = 512,
						  initializer    = initializer, 
						  padding        = "SAME", 
						  scope          = "conv3x3")
						  
			net = batch_norm(net, is_training = is_training)
							  
			net = tf.nn.relu(features = net, name = "ReLU")

		with tf.variable_scope("layer11"):
			net = conv2D( net,
						  kernel_size    = 3,
						  strides        = 1, 
						  output_channel = 512,
						  initializer    = initializer, 
						  padding        = "SAME", 
						  scope          = "conv3x3")
						  
			net = batch_norm(net, is_training = is_training)
							  
			net = tf.nn.relu(features = net, name = "ReLU")

		with tf.variable_scope("layer12"): # With Pooling
			net = conv2D( net,
						  kernel_size    = 3,
						  strides        = 1, 
						  output_channel = 512,
						  initializer    = initializer, 
						  padding        = "SAME", 
						  scope          = "conv3x3")
						  
			net = batch_norm(net, is_training = is_training)
							  
			net = tf.nn.relu(features = net, name = "ReLU")

			net, indices4, unpool_shape4 = max_pooling(net, kernel_size = 2, strides = 2, padding = "SAME")
		
		with tf.variable_scope("layer13"): # With Unpooling
			net = max_unpooling(net, output_shape = unpool_shape4, indices = indices4)
			
			net = conv2D( net,
						  kernel_size    = 3,
						  strides        = 1, 
						  output_channel = 512,
						  initializer    = initializer, 
						  padding        = "SAME", 
						  scope          = "conv3x3")
						  
			net = batch_norm(net, is_training = is_training)
							  
			net = tf.nn.relu(features = net, name = "ReLU")										
				
		with tf.variable_scope("layer14"):
			net = conv2D( net,
						  kernel_size    = 3,
						  strides        = 1, 
						  output_channel = 512,
						  initializer    = initializer, 
						  padding        = "SAME", 
						  scope          = "conv3x3")
						  
			net = batch_norm(net, is_training = is_training)
							  
			net = tf.nn.relu(features = net, name = "ReLU")
		
		with tf.variable_scope("layer15"):
			net = conv2D( net,
						  kernel_size    = 3,
						  strides        = 1, 
						  output_channel = 512,
						  initializer    = initializer, 
						  padding        = "SAME", 
						  scope          = "conv3x3")
						  
			net = batch_norm(net, is_training = is_training)
							  
			net = tf.nn.relu(features = net, name = "ReLU")
		
		with tf.variable_scope("layer16"): # With Unpooling
			net = max_unpooling(net, output_shape = unpool_shape3, indices = indices3)
			
			net = conv2D( net,
						  kernel_size    = 3,
						  strides        = 1, 
						  output_channel = 512,
						  initializer    = initializer, 
						  padding        = "SAME", 
						  scope          = "conv3x3")
						  
			net = batch_norm(net, is_training = is_training)
							  
			net = tf.nn.relu(features = net, name = "ReLU")										
				
		with tf.variable_scope("layer17"):
			net = conv2D( net,
						  kernel_size    = 3,
						  strides        = 1, 
						  output_channel = 512,
						  initializer    = initializer, 
						  padding        = "SAME", 
						  scope          = "conv3x3")
						  
			net = batch_norm(net, is_training = is_training)
							  
			net = tf.nn.relu(features = net, name = "ReLU")
		
		with tf.variable_scope("layer18"):
			net = conv2D( net,
						  kernel_size    = 3,
						  strides        = 1, 
						  output_channel = 256,
						  initializer    = initializer, 
						  padding        = "SAME", 
						  scope          = "conv3x3")
						  
			net = batch_norm(net, is_training = is_training)
							  
			net = tf.nn.relu(features = net, name = "ReLU")
		
		with tf.variable_scope("layer19"): # With Unpooling
			net = max_unpooling(net, output_shape = unpool_shape2, indices = indices2)
			
			net = conv2D( net,
						  kernel_size    = 3,
						  strides        = 1, 
						  output_channel = 256,
						  initializer    = initializer, 
						  padding        = "SAME", 
						  scope          = "conv3x3")
						  
			net = batch_norm(net, is_training = is_training)
							  
			net = tf.nn.relu(features = net, name = "ReLU")										

		with tf.variable_scope("layer20"):
			net = conv2D( net,
						  kernel_size    = 3,
						  strides        = 1, 
						  output_channel = 256,
						  initializer    = initializer, 
						  padding        = "SAME", 
						  scope          = "conv3x3")
						  
			net = batch_norm(net, is_training = is_training)
							  
			net = tf.nn.relu(features = net, name = "ReLU")
			
		with tf.variable_scope("layer21"):
			net = conv2D( net,
						  kernel_size    = 3,
						  strides        = 1, 
						  output_channel = 128,
						  initializer    = initializer, 
						  padding        = "SAME", 
						  scope          = "conv3x3")
						  
			net = batch_norm(net, is_training = is_training)
							  
			net = tf.nn.relu(features = net, name = "ReLU")

		with tf.variable_scope("layer22"): # With Unpooling
			net = max_unpooling(net, output_shape = unpool_shape1, indices = indices1)
			
			net = conv2D( net,
						  kernel_size    = 3,
						  strides        = 1, 
						  output_channel = 128,
						  initializer    = initializer, 
						  padding        = "SAME", 
						  scope          = "conv3x3")
						  
			net = batch_norm(net, is_training = is_training)
							  
			net = tf.nn.relu(features = net, name = "ReLU")										

		with tf.variable_scope("layer23"):
			net = conv2D( net,
						  kernel_size    = 3,
						  strides        = 1, 
						  output_channel = 64,
						  initializer    = initializer, 
						  padding        = "SAME", 
						  scope          = "conv3x3")
						  
			net = batch_norm(net, is_training = is_training)
							  
			net = tf.nn.relu(features = net, name = "ReLU")

		with tf.variable_scope("layer24"): # With Unpooling
			net = max_unpooling(net, output_shape = unpool_shape0, indices = indices0)
			
			net = conv2D( net,
						  kernel_size    = 3,
						  strides        = 1, 
						  output_channel = 64,
						  initializer    = initializer, 
						  padding        = "SAME", 
						  scope          = "conv3x3")
						  
			net = batch_norm(net, is_training = is_training)
							  
			net = tf.nn.relu(features = net, name = "ReLU")										
			
			net = tf.cond(is_training, lambda: tf.layers.dropout(net, 0.0), lambda: net)
			
		with tf.variable_scope("layer25"):
			net = conv2D( net,
						  kernel_size    = 3,
						  strides        = 1, 
						  output_channel = CLASS_NUM,
						  initializer    = initializer, 
						  padding        = "SAME", 
						  scope          = "conv3x3")
						  
			net = tf.nn.relu(features = net, name = "ReLU")
	return net


def your_own_model(
	net, 
	):

	return net


#============#
#    main    #
#============#
def main(argv=None):

	assert (IS_TRAINING or IS_TESTING) is True, "Must one of the IS_TRAINING & IS_TESTING be True!"

	#---------------#
	#    Dataset    #
	#---------------#
	if IS_TRAINING:
		#******************#
		#    train data    #
		#******************#
		print("Loading training data ... ")
		# read index
		train_data_index   = open(Dataset_Path + '/train.txt'     , 'r').read().splitlines()
		train_target_index = open(Dataset_Path + '/trainannot.txt', 'r').read().splitlines()
		# read images
		train_data, train_target = CamVid_dataset_parser(train_data_index, train_target_index)
		# one-hot target
		train_target = one_hot(target = train_target, class_num = CLASS_NUM)
		print("\033[0;32mTrain Data Number\033[0m = {}" .format( np.shape(train_data)[0]   ))
		print("\033[0;32mTrain Data Shape \033[0m = {}" .format( np.shape(train_data)[1:4] )) # [Height, Width, Depth]
		
		#****************#
		#    val data    #
		#****************#
		print("Loading validation data ... ")
		# read index
		val_data_index     = open(Dataset_Path + '/val.txt'       , 'r').read().splitlines()
		val_target_index   = open(Dataset_Path + '/valannot.txt'  , 'r').read().splitlines()
		# read images
		val_data, val_target = CamVid_dataset_parser(val_data_index, val_target_index)
		# one-hot target
		val_target = one_hot(target = val_target, class_num = CLASS_NUM)
		print("\033[0;32mVal Data Number\033[0m   = {}" .format( np.shape(val_data)[0]   ))
		print("\033[0;32mVal Data Shape \033[0m   = {}" .format( np.shape(val_data)[1:4] )) # [Height, Width, Depth]
	
	if IS_TESTING:
		#*****************#
		#    test data    #
		#*****************#
		print("Loading testing data ... ")
		# read index
		test_data_index    = open(Dataset_Path + '/test.txt'      , 'r').read().splitlines()
		test_target_index  = open(Dataset_Path + '/testannot.txt' , 'r').read().splitlines()
		# read images
		test_data, test_target = CamVid_dataset_parser(test_data_index, test_target_index)
		# one-hot target
		test_target = one_hot(target = test_target, class_num = CLASS_NUM)
		print("\033[0;32mTest Data Number\033[0m  = {}" .format( np.shape(test_data)[0]   ))
		print("\033[0;32mTest Data Shape \033[0m  = {}" .format( np.shape(test_data)[1:4] )) # [Height, Width, Depth]
	
	#-------------------#
	#    Placeholder    #
	#-------------------#
	try:
		data_shape = np.shape(train_data)
	except:
		data_shape = np.shape(test_data)
	
	xs = tf.placeholder(dtype = tf.float32, shape = [BATCH_SIZE, data_shape[1], data_shape[2], data_shape[3]])
	ys = tf.placeholder(dtype = tf.float32, shape = [BATCH_SIZE, data_shape[1], data_shape[2], CLASS_NUM])
	lr = tf.placeholder(dtype = tf.float32)
	is_training = tf.placeholder(dtype = tf.bool)
	
	#-------------#
	#    Model    #
	#-------------#
	net = xs
	prediction = SegNet( net, 
	                     is_training, 
						 initializer = tf.contrib.layers.variance_scaling_initializer(), 
						 class_num   = CLASS_NUM, 
						 scope       = "SegNet")
	
	#---------------------#
	#    Loss Function    #
	#---------------------#
	# (Optional) You can choose another loss function
	# i.e. Cross Entropy : https://www.tensorflow.org/api_docs/python/tf/nn/sparse_softmax_cross_entropy_with_logits
	cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels = tf.argmax(ys, -1), logits = prediction)
	
	# L2_norm
	# (Optional)
	weights_collection = tf.get_collection("weights")
	l2_norm   = tf.reduce_mean(tf.stack([tf.nn.l2_loss(x) for x in weights_collection]))
	l2_lambda = tf.constant(0.9)
	l2_norm   = tf.multiply(l2_lambda, l2_norm)
	
	
	# Loss Function
	loss = cross_entropy
	"""
	loss = tf.add(loss, l2_norm)
	"""
	
	#-----------------------------------#
	#    Weight Optimization Strategy   #
	#-----------------------------------#
	# (Optional) You can choose another optimizer
	# i.e. SGD : https://www.tensorflow.org/api_docs/python/tf/train/GradientDescentOptimizer
	opt = tf.train.GradientDescentOptimizer(learning_rate = lr)
	#opt = tf.train.MomentumOptimizer(learning_rate = lr, momentum = 0.2)
	#opt = tf.train.AdamOptimizer(learning_rate = lr)
	gra_and_var = opt.compute_gradients(loss)
	train_step  = opt.apply_gradients(gra_and_var)
	
	#-------------#
	#    Saver    #
	#-------------#
	saver = tf.train.Saver()
	
	#---------------#
	#    Session    #
	#---------------#
	if IS_TRAINING:
		with tf.Session() as sess:
			# Initial all tensor variables
			init = tf.global_variables_initializer()
			sess.run(init)
			
			print("Training ... ")
			for epoch in range(EPOCH):
				tStart = time.time()
				print("\r\033[0;33mEpoch {}\033[0m" .format(epoch), end="")
				
				train_data_   = train_data
				train_target_ = train_target
				
				# Shuffle Data
				# (Optional)
				"""
				train_data_, train_target_ = shuffle_data(train_data, train_target)
				"""
				
				#----------------#
				#    Training    #
				#----------------#
				total_iter = int(np.shape(train_data_)[0] / BATCH_SIZE)
				total_accuracy_top1 = 0
				for iter in range(total_iter):
					# Batch Data
					batch_xs = train_data_  [ iter*BATCH_SIZE : (iter+1)*BATCH_SIZE ]
					batch_ys = train_target_[ iter*BATCH_SIZE : (iter+1)*BATCH_SIZE ]
					
					# Run Optimization
					_, y_pre, Loss = sess.run([train_step, prediction, loss],
											feed_dict={ xs: batch_xs,
														ys: batch_ys,
														lr: 1e-7,												
														is_training: True})
					# Calculate Training Accuracy
					prediction_top1 = np.argmax(y_pre, axis=-1)
					correct_prediction_top1 = np.equal(prediction_top1, np.argmax(batch_ys, -1))
					accuracy_top1 = np.mean(correct_prediction_top1.astype(float))
	
					# Show Each Class Training Accuracy
					# (Optional)
					"""
					per_class_accuracy(y_pre, batch_ys)
					"""
					
					total_accuracy_top1 = total_accuracy_top1 + accuracy_top1
					
				total_accuracy_top1 = total_accuracy_top1 / total_iter
				
				tEnd = time.time()
				print(" (cost {TIME} sec)" .format(TIME = tEnd - tStart))
				print("\033[0;32mTraining Accuracy\033[0m = {}" .format(total_accuracy_top1))
				
				#------------------#
				#    Validation    #
				#------------------#
				# (Optional)
				"""
				if (epoch+1)%10 == 0:
					val_result, val_accuracy = compute_accuracy( xs, ys, is_training, prediction, 
													             v_xs       = val_data,
													             v_ys       = val_target, 
													             batch_size = BATCH_SIZE, 
													             sess       = sess)
					print("\033[0;32mValidation Accuracy\033[0m = {}" .format(val_accuracy))
				"""
				
			#----------------------------#
			#    Save Trained Weights    #
			#----------------------------#
			print("Saving Trained Weights ... ")
			save_path = saver.save(sess, "HW5_example.ckpt")
			print(save_path)
	
	
	if IS_TESTING:
		with tf.Session() as sess:
			# Initial all tensor variables
			init = tf.global_variables_initializer()
			sess.run(init)
			
			# Loading trained weights
			print("Loading trained weights ...")
			saver.restore(sess, "HW5_example.ckpt")
			
			print("Testing ... ")
			test_result, test_accuracy = compute_accuracy( xs, ys, is_training, prediction, 
											               v_xs       = test_data,
											               v_ys       = test_target, 
											               batch_size = BATCH_SIZE, 
											               sess       = sess)
			print("\033[0;32mTesting Accuracy\033[0m = {}" .format(test_accuracy))
			
			# Save the test result to the image
			# If you use this code, you can see the images at 
			# ->  /nets/CamVid_Y_pre/testannot/
			# (Optional)
			"""
			save_CamVid_result_as_image(
				result     = test_result, 
				path       = 'CamVid_Y_pre',
				file_index = test_target_index)
			"""
			
if __name__ == "__main__":
	tf.app.run(main, sys.argv)
	
	
	
	
	
	
