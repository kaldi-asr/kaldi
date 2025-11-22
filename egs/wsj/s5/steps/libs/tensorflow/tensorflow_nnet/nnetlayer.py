# Copyright 2016    Vincent Renkens
##@package nnetlayer
# contains neural network layers 

import tensorflow as tf

##This class defines a fully connected feed forward layer
class FFLayer(object):

	##FFLayer constructor, defines the variables
	#
	#@param output_dim output dimension of the layer
	#@param activation the activation function
	#@param weights_std the standart deviation of the weights by default the inverse square root of the input dimension is taken
	def __init__(self, output_dim, activation, weights_std=None):
						
		#save the parameters
		self.output_dim = output_dim
		self.activation = activation
		self.weights_std = weights_std
		
	##Do the forward computation
	#
	#@param inputs the input to the layer
	#@param is_training is_training whether or not the network is in training mode
	#@param reuse wheter or not the variables in the network should be reused
	#@param scope the variable scope of the layer
	#
	#@return the output of the layer and the training output of the layer
	def __call__(self, inputs, is_training = False, reuse = False, scope = None):
			
		with tf.variable_scope(scope or type(self).__name__, reuse=reuse):
			with tf.variable_scope('parameters', reuse=reuse):
				weights = tf.get_variable('weights', [inputs.get_shape()[1], self.output_dim], initializer=tf.random_normal_initializer(stddev=self.weights_std if self.weights_std is not None else 1/int(inputs.get_shape()[1])**0.5))
				biases = tf.get_variable('biases',  [self.output_dim], initializer=tf.constant_initializer(0))
			
			#apply weights and biases
			with tf.variable_scope('linear', reuse=reuse):
				linear = tf.matmul(inputs, weights) + biases
				
			#apply activation function	
			with tf.variable_scope('activation', reuse=reuse):
				outputs = self.activation(linear, is_training, reuse)

		return outputs
	
