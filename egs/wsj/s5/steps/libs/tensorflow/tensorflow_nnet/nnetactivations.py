# Copyright 2016    Vincent Renkens
##@package nnetactivations
# contains activation functions

import tensorflow as tf

##a class for activation functions
class activation(object):
	##apply the activation function
	#
	#@param inputs the inputs to the activation function
	#@param is_training is_training whether or not the network is in training mode
	#@param reuse wheter or not the variables in the network should be reused
	#
	#@return the output to the activation function
	def __call__(self, inputs, is_training = False, reuse = False):
		
		if self.activation is not None:
			#apply the wrapped activation
			activations = self.activation(inputs, is_training, reuse)
		else:
			activations = inputs
		
		#add own computation
		activation = self._apply_func(activations, is_training, reuse)
		
		return activation
		
	##apply own functionality
	#
	#@param activations the ioutputs to the wrapped activation function
	#@param is_training is_training whether or not the network is in training mode
	#@param reuse wheter or not the variables in the network should be reused
	#
	#@return the output to the activation function
	def _apply_func(self, activations, is_training, reuse):
		raise NotImplementedError("Abstract method")
	
		
##a wrapper for an activation function that will add a tf activation function
class Tf_wrapper(activation):
	##the Tf_wrapper constructor
	#
	#@param activation the activation function being wrapped
	#@param the tensorflow activation function that is wrapping
	def __init__(self, activation, tf_activation):
		self.activation = activation
		self.tf_activation = tf_activation
		
	##apply own functionality
	#
	#@param activations the ioutputs to the wrapped activation function
	#@param is_training is_training whether or not the network is in training mode
	#@param reuse wheter or not the variables in the network should be reused
	#
	#@return the output to the activation function
	def _apply_func(self, activations, is_training, reuse):
		
		return self.tf_activation(activations)
		

##a wrapper for an activation function that will add l2 normalisation
class L2_wrapper(activation):
	##the L2_wrapper constructor
	#
	#@param activation the activation function being wrapped
	def __init__(self, activation):
		self.activation = activation
		
	##apply own functionality
	#
	#@param activations the ioutputs to the wrapped activation function
	#@param is_training is_training whether or not the network is in training mode
	#@param reuse wheter or not the variables in the network should be reused
	#
	#@return the output to the activation function
	def _apply_func(self, activations, is_training, reuse):
		
		with tf.variable_scope('l2_norm', reuse=reuse):
			#compute the mean squared value
			sig = tf.reduce_mean(tf.square(activations), 1, keep_dims=True)
			
			#divide the input by the mean squared value
			normalized = activations/sig
			
			#if the mean squared value is larger then one select the normalized value otherwise select the unnormalised one
			return tf.select(tf.greater(tf.reshape(sig, [-1]), 1), normalized, activations)
			
## a wrapper for an activation function that will add dropout
class Dropout_wrapper(activation):
	##the Dropout_wrapper constructor
	#
	#@param activation the activation function being wrapped
	#@param dopout the dropout rate, has to be a value in (0:1]
	def __init__(self, activation, dropout):
		self.activation = activation
		assert(dropout > 0 and dropout <= 1)
		self.dropout = dropout
		
	##apply own functionality
	#
	#@param activations the ioutputs to the wrapped activation function
	#@param is_training is_training whether or not the network is in training mode
	#@param reuse wheter or not the variables in the network should be reused
	#
	#@return the output to the activation function
	def _apply_func(self, activations, is_training, reuse):
		
		if is_training:
			return tf.nn.dropout(activations, self.dropout)
		else:
			return activations
		

## a wrapper for an activation function that will add batch normalisation
class Batchnorm_wrapper(activation):
	##the Batchnorm_wrapper constructor
	#
	#@param activation the activation function being wrapped
	def __init__(self, activation):
		self.activation = activation
		
	##apply own functionality
	#
	#@param activations the ioutputs to the wrapped activation function
	#@param is_training is_training whether or not the network is in training mode
	#@param reuse wheter or not the variables in the network should be reused
	#
	#@return the output to the activation function
	def _apply_func(self, activations, is_training, reuse):
		return tf.contrib.layers.batch_norm(activations, is_training=is_training, reuse=reuse, scope='batch_norm')
		
			
