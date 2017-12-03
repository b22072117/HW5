from __future__ import print_function
import tensorflow as tf
import numpy as np
import scipy.ndimage
from scipy import misc

import sys
import os
import time
import pdb

import utils
import model

#========================#
#    Global Parameter    #
#========================#
Student_ID = sys.argv[1]
model_call = getattr(model, Student_ID) # using string to call function
BATCH_SIZE = 4
CLASS_NUM  = 12
EPOCH = 50
Dataset_Path = '../datasets/CamVid'


#============#
#    main    #
#============#
def main(argv=None):
	#---------------#
	#    Dataset    #
	#---------------#
	#******************#
	#    train data    #
	#******************#
	print("Loading training data ... ")
	# read index
	train_data_index   = open(Dataset_Path + '/train.txt'     , 'r').read().splitlines()
	train_target_index = open(Dataset_Path + '/trainannot.txt', 'r').read().splitlines()
	# read images
	train_data, train_target = utils.CamVid_dataset_parser(Dataset_Path, train_data_index, train_target_index)
	# one-hot target
	train_target = utils.one_hot(target = train_target, class_num = CLASS_NUM)
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
	val_data, val_target = utils.CamVid_dataset_parser(Dataset_Path, val_data_index, val_target_index)
	# one-hot target
	val_target = utils.one_hot(target = val_target, class_num = CLASS_NUM)
	print("\033[0;32mVal Data Number\033[0m   = {}" .format( np.shape(val_data)[0]   ))
	print("\033[0;32mVal Data Shape \033[0m   = {}" .format( np.shape(val_data)[1:4] )) # [Height, Width, Depth]
	
		
	#-------------------#
	#    Placeholder    #
	#-------------------#
	data_shape = np.shape(train_data)
	xs = tf.placeholder(dtype = tf.float32, shape = [BATCH_SIZE, data_shape[1], data_shape[2], data_shape[3]])
	ys = tf.placeholder(dtype = tf.float32, shape = [BATCH_SIZE, data_shape[1], data_shape[2], CLASS_NUM])
	lr = tf.placeholder(dtype = tf.float32)
	is_training = tf.placeholder(dtype = tf.bool)
	
	#-------------#
	#    Model    #
	#-------------#
	net = xs
	# Call your model here
	prediction = model_call( net, 
	                         is_training, 
						     initializer = tf.contrib.layers.variance_scaling_initializer(), 
						     class_num   = CLASS_NUM, 
						     scope       = Student_ID)
	
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
	gra_and_var = opt.compute_gradients(loss)
	train_step  = opt.apply_gradients(gra_and_var)
	
	#-------------#
	#    Saver    #
	#-------------#
	saver = tf.train.Saver()
	
	#---------------#
	#    Session    #
	#---------------#
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
			train_data_, train_target_ = utils.shuffle_data(train_data, train_target)
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
				utils.per_class_accuracy(y_pre, batch_ys)
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
				val_result, val_accuracy = utils.compute_accuracy( xs, ys, is_training, prediction, 
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
		save_path = saver.save(sess, Student_ID + ".ckpt")
		print(save_path)
			
if __name__ == "__main__":
	tf.app.run(main, sys.argv)
	
	
	
	
	
	
