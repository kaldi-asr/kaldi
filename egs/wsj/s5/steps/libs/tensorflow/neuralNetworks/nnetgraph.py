# Copyright 2016    Vincent Renkens
##@package nnetgraph
# contains the functionality to create neural network graphs and train/test it

import tensorflow as tf
import numpy as np
from nnetlayer import FFLayer
import nnetactivations

import pdb

##This an abstrace class defining a neural net
class NnetGraph(object):
	
	##Add the neural net variables and operations to the graph
	#
	#@param inputs the inputs to the neural network
	#@param scope the name scope 
	#
	#@return logits used for training, logits used for testing, a saver object and a dictionary of control operations (may be empty)
	def __call__(self, inputs):
		raise NotImplementedError("Abstract method")
		
	@property
	def output_dim(self):
		return self._output_dim

		
	
		
##This class is a graph for feedforward fully connected neural nets.
class DNN(NnetGraph):
	
	## DNN constructor
	#
	#@param output_dim the DNN output dimension
	#@param num_layers number of hidden layers
	#@param num_units number of hidden units
	#@param activation the activation function
	#@param layerwise_init if True the layers will be added one by one, otherwise all layers will be added to the network in the beginning
	def __init__(self, output_dim, num_layers, num_units, activation, layerwise_init = True):
		
		#save all the DNN properties
		self._output_dim = output_dim
		self.num_layers = num_layers
		self.num_units = num_units
		self.activation = activation
		self.layerwise_init = layerwise_init
	
	##Add the DNN variables and operations to the graph
	#
	#@param inputs the inputs to the neural network
	#@param is_training is_training whether or not the network is in training mode
	#@param reuse wheter or not the variables in the network should be reused
	#@param scope the name scope 
	#
	#@return output logits, a saver object and a list of control operations (add:add layer, init:initialise output layer)
	def __call__(self, inputs, is_training = False, reuse = False, scope = None):
			
		with tf.variable_scope(scope or type(self).__name__, reuse=reuse):
								
		 	#input layer
		 	layer = FFLayer(self.num_units, self.activation)
		 		
	 		#output layer
	 		outlayer = FFLayer(self.output_dim, nnetactivations.Tf_wrapper(None, lambda(x): x), 0)
	 		
			#do the forward computation with dropout
	 		
	 		trainactivations = [None]*self.num_layers
	 		activations = [None]*self.num_layers
			activations[0] = layer(inputs, is_training, reuse, 'layer0')
			for l in range(1,self.num_layers):
				activations[l] = layer(activations[l-1], is_training, reuse, 'layer' + str(l))
	 		
	 		if self.layerwise_init:
	 		
	 			#variable that determines how many layers are initialised in the neural net
				initialisedlayers = tf.get_variable('initialisedlayers', [], initializer=tf.constant_initializer(0), trainable=False, dtype=tf.int32)
		
				#operation to increment the number of layers
				addLayerOp = initialisedlayers.assign_add(1).op
	 		
				#compute the logits by selecting the activations at the layer that has last been added to the network, this is used for layer by layer initialisation
				logits = tf.case([(tf.equal(initialisedlayers, tf.constant(l)), CallableTensor(activations[l])) for l in range(len(activations))], default=CallableTensor(activations[-1]),exclusive=True,name = 'layerSelector')
			
				logits.set_shape([None, self.num_units])
				
			else:
				logits = activations[-1]
				
			logits = outlayer(logits, is_training, reuse, 'layer' + str(self.num_layers))
			
			
			if self.layerwise_init:
		 		#operation to initialise the final layer
		 		initLastLayerOp = tf.initialize_variables(tf.get_collection(tf.GraphKeys.VARIABLES, scope=tf.get_variable_scope().name + '/layer' + str(self.num_layers)))
		 		
		 		control_ops = {'add':addLayerOp, 'init':initLastLayerOp}
	 		else:
	 			control_ops = None
		
			#create a saver 
			saver = tf.train.Saver()
			
		return logits, saver, control_ops

##Class for the decoding environment for a neural net graph
class NnetDecoder(object):
	##NnetDecoder constructor, creates the decoding graph
	#
	#@param nnetGraph an nnetgraph object for the neural net that will be used for decoding
	#@param input_dim the input dimension to the nnnetgraph
	def __init__(self, nnetGraph, input_dim):

		self.graph = tf.Graph()
		
		with self.graph.as_default():
		
			#create the inputs placeholder
			self.inputs = tf.placeholder(tf.float32, shape = [None, input_dim], name = 'inputs')
		
			#create the decoding graph
			logits, self.saver, _ = nnetGraph(self.inputs, is_training = False, reuse = False)
			
			#compute the outputs
			self.outputs = tf.nn.softmax(logits)
	
		#specify that the graph can no longer be modified after this point
		self.graph.finalize()
	
	##decode using the neural net
	#
	#@param inputs the inputs to the graph as a NxF numpy array where N is the number of frames and F is the input feature dimension
	#
	#@return an NxO numpy array where N is the number of frames and O is the neural net output dimension
	def __call__(self, inputs):
		return self.outputs.eval(feed_dict = {self.inputs:inputs})
	
	##load the saved neural net
	#
	#@param filename location where the neural net is saved
	def restore(self, filename):
		self.saver.restore(tf.get_default_session(), filename)
	
 					
##Class for the training environment for a neural net graph
class NnetTrainer(object):

	#NnetTrainer constructor, creates the training graph
	#
	#@param nnetgraph an nnetgraph object for the neural net that will be used for decoding
	#@param input_dim the input dimension to the nnnetgraph
	#@param init_learning_rate the initial learning rate
	#@param learning_rate_decay the parameter for exponential learning rate decay
	#@param num_steps the total number of steps that will be taken
	#@param numframes_per_batch determines how many frames are processed at a time to limit memory usage
	def __init__(self, nnetGraph, input_dim, init_learning_rate, learning_rate_decay, num_steps, numframes_per_batch):
	
		self.numframes_per_batch = numframes_per_batch
	
		#create the graph
		self.graph = tf.Graph()
		
		#define the placeholders in the graph
		with self.graph.as_default():
			
			#create the inputs placeholder
			self.inputs = tf.placeholder(tf.float32, shape = [None, input_dim], name = 'inputs')
			
			#reference labels
			self.targets = tf.placeholder(tf.float32, shape = [None, nnetGraph.output_dim], name = 'targets')
			
			#input for the total number of frames that are used in the batch
			self.num_frames = tf.placeholder(tf.float32, shape = [], name = 'num_frames')
			
			#compute the training outputs of the nnetgraph
			trainlogits, self.modelsaver, self.control_ops = nnetGraph(self.inputs, is_training = True, reuse = False)
			
			#compute the validation output of the nnetgraph 
			logits, _, _ = nnetGraph(self.inputs, is_training = False, reuse = True, scope = 'DNN')
			
			#get a list of trainable variables in the decoder graph
			params = tf.trainable_variables()

			#add the variables and operations to the graph that are used for training
			
			#total number of steps
			Nsteps = tf.constant(num_steps, dtype = tf.int32, name = 'num_steps')
			
			#the total loss of the entire batch
			batch_loss = tf.get_variable('batch_loss', [], dtype=tf.float32, initializer=tf.constant_initializer(0), trainable=False)
			
			with tf.variable_scope('train_variables'):	

				#the amount of steps already taken
				self.global_step = tf.get_variable('global_step', [], dtype=tf.int32, initializer=tf.constant_initializer(0), trainable=False) 
	
				#a variable to scale the learning rate (used to reduce the learning rate in case validation performance drops)
				learning_rate_fact = tf.get_variable('learning_rate_fact', [], initializer=tf.constant_initializer(1.0), trainable=False)
				
				#compute the learning rate with exponential decay and scale with the learning rate factor
				learning_rate = tf.train.exponential_decay(init_learning_rate, self.global_step, Nsteps, learning_rate_decay) * learning_rate_fact
				
				#create the optimizer
				optimizer = tf.train.AdamOptimizer(learning_rate)
			
			#for every parameter create a variable that holds its gradients
			with tf.variable_scope('gradients'):
				grads = [tf.get_variable(param.op.name, param.get_shape().as_list(), initializer=tf.constant_initializer(0), trainable=False) for param in params]
				
			with tf.name_scope('train'):
				#compute the training loss
				loss = tf.reduce_sum(self.computeLoss(self.targets, trainlogits))
			
				#operation to half the learning rate
				self.halveLearningRateOp = learning_rate_fact.assign(learning_rate_fact/2).op
				
				#create an operation to initialise the gradients
				self.initgrads = tf.initialize_variables(grads)
				
				#the operation to initialise the batch loss
				self.initloss = batch_loss.initializer
				
				#compute the gradients of the batch
				batchgrads = tf.gradients(loss, params)
				
				#create an operation to update the batch loss
				self.updateLoss = batch_loss.assign_add(loss).op
				
				#create an operation to update the gradients, the batch_loss and do all other update ops
				update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
				self.updateGradientsOp = tf.group(*([grads[p].assign_add(batchgrads[p]) for p in range(len(grads)) if batchgrads[p] is not None] + [self.updateLoss] + update_ops), name='update_gradients')
				
				#create an operation to apply the gradients
				meangrads = [tf.div(grad,self.num_frames, name=grad.op.name) for grad in grads]
				self.applyGradientsOp = optimizer.apply_gradients([(meangrads[p], params[p]) for p in range(len(meangrads))], global_step=self.global_step, name='apply_gradients')
			
			with tf.name_scope('valid'):
				#compute the validation loss
				validLoss = tf.reduce_sum(self.computeLoss(self.targets, logits))
				
				#operation to update the validation loss
				self.updateValidLoss = batch_loss.assign_add(validLoss).op
			
			#operation to compute the average loss in the batch
			self.average_loss = batch_loss/self.num_frames
			
			# add an operation to initialise all the variables in the graph
			self.initop = tf.initialize_all_variables()
			
			#saver for the training variables
			self.saver = tf.train.Saver(tf.get_collection(tf.GraphKeys.VARIABLES, scope='train_variables'))
			
			#create the summaries for visualisation
			self.summary = tf.merge_summary([tf.histogram_summary(val.name, val) for val in params+meangrads] + [tf.scalar_summary('loss', self.average_loss)])
			
			
		#specify that the graph can no longer be modified after this point
		self.graph.finalize()
		
		#start without visualisation
		self.summarywriter = None
		
	##Creates the operation to compute the cross-enthropy loss for every input frame (if you want to have a different loss function, overwrite this method)
	#
	#@param targets a NxO tensor containing the reference targets where N is the number of frames and O is the neural net output dimension
	#@param logits a NxO tensor containing the neural network output logits where N is the number of frames and O is the neural net output dimension
	#
	#@return an N-dimensional tensor containing the losses for all the input frames where N is the number of frames
	def computeLoss(self, targets, logits):
		return tf.nn.softmax_cross_entropy_with_logits(logits, targets, name='loss')
		
	##Initialize all the variables in the graph
	def initialize(self):
		self.initop.run()
		
	##open a summarywriter for visualisation and add the graph
	#
	#@param logdir directory where the summaries will be written
	def startVisualization(self, logdir):
		self.summarywriter = tf.train.SummaryWriter(logdir=logdir, graph=self.graph)
	
	##update the neural model with a batch or training data
	#
	#@param inputs the inputs to the neural net, this should be a NxF numpy array where N is the number of frames in the batch and F is the feature dimension
	#@param targets the one-hot encoded targets for neural nnet, this should be an NxO matrix where O is the output dimension of the neural net
	#
	#@return the loss at this step
	def update(self, inputs, targets):
		
		#if numframes_per_batch is not set just process the entire batch
		if self.numframes_per_batch==-1 or self.numframes_per_batch>inputs.shape[0]:
			numframes_per_batch = inputs.shape[0]
		else:
			numframes_per_batch = self.numframes_per_batch
				
		#feed in the batches one by one and accumulate the gradients and loss
		for k in range(int(inputs.shape[0]/numframes_per_batch) + int(inputs.shape[0]%numframes_per_batch > 0)):
			batchInputs = inputs[k*numframes_per_batch:min((k+1)*numframes_per_batch, inputs.shape[0]), :]
			batchTargets = targets[k*numframes_per_batch:min((k+1)*numframes_per_batch, inputs.shape[0]), :]
			self.updateGradientsOp.run(feed_dict = {self.inputs:batchInputs, self.targets:batchTargets})
			
		#apply the accumulated gradients to update the model parameters and evaluate the loss
		if self.summarywriter is not None:
			[loss, summary, _] = tf.get_default_session().run([self.average_loss, self.summary, self.applyGradientsOp], feed_dict = {self.num_frames:inputs.shape[0]})
			self.summarywriter.add_summary(summary, global_step=self.global_step.eval())
		else:
			[loss, _] = tf.get_default_session().run([self.average_loss, self.applyGradientsOp], feed_dict = {self.num_frames:inputs.shape[0]})
			
		
		#reinitialize the gradients and the loss
		self.initgrads.run()
		self.initloss.run()
		
		return loss
		

	##Evaluate the performance of the neural net
	#
	#@param inputs the inputs to the neural net, this should be a NxF numpy array where N is the number of frames in the batch and F is the feature dimension
	#@param targets the one-hot encoded targets for neural nnet, this should be an NxO matrix where O is the output dimension of the neural net
	#
	#@return the loss of the batch
	def evaluate(self, inputs, targets):
		
		if inputs is None or targets is None:
			return None
	
		#if numframes_per_batch is not set just process the entire batch
		if self.numframes_per_batch==-1 or self.numframes_per_batch>inputs.shape[0]:
			numframes_per_batch = inputs.shape[0]
		else:
			numframes_per_batch = self.numframes_per_batch
					
		#feed in the batches one by one and accumulate the loss
		for k in range(int(inputs.shape[0]/self.numframes_per_batch) + int(inputs.shape[0]%self.numframes_per_batch > 0)):
			batchInputs = inputs[k*self.numframes_per_batch:min((k+1)*self.numframes_per_batch, inputs.shape[0]), :]
			batchTargets = targets[k*self.numframes_per_batch:min((k+1)*self.numframes_per_batch, inputs.shape[0]), :]
			self.updateValidLoss.run(feed_dict = {self.inputs:batchInputs, self.targets:batchTargets})
			
		#get the loss
		loss = self.average_loss.eval(feed_dict = {self.num_frames:inputs.shape[0]})
		
		#reinitialize the loss
		self.initloss.run()
		
		return loss
		
			
	##halve the learning rate
	def halve_learning_rate(self):
		self.halveLearningRateOp.run()
	
	##Save the model
	#
	#@param filename path to the model file
	def saveModel(self, filename):
		self.modelsaver.save(tf.get_default_session(), filename)
		
	##Load the model
	#
	#@param filename path where the model will be saved
	def restoreModel(self, filename):
		self.modelsaver.restore(tf.get_default_session(), filename)
		
	##Save the training progress (including the model)
	#
	#@param filename path where the model will be saved
	def saveTrainer(self, filename):
		self.modelsaver.save(tf.get_default_session(), filename)
		self.saver.save(tf.get_default_session(), filename + '_trainvars')
		
	##Load the training progress (including the model)
	#
	#@param filename path where the model will be saved
	def restoreTrainer(self, filename):
		self.modelsaver.restore(tf.get_default_session(), filename)
		self.saver.restore(tf.get_default_session(), filename + '_trainvars')

##A class for a tensor that is callable		
class CallableTensor:
	##CallableTensor constructor
	#
	#@param tensor a tensor
	def __init__(self, tensor):
		self.tensor = tensor
	##get the tensor
	#
	#@return the tensor
	def __call__(self):
		return self.tensor
			

