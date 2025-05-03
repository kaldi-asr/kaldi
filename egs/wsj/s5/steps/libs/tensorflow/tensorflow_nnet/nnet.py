# Copyright 2016    Vincent Renkens
##@package nnet
# contains the functionality for a Kaldi style neural network

import numpy as np
from six.moves import cPickle as pickle
from six.moves import range
import tensorflow as tf
import gzip
import shutil
import os
import sys
sys.path.append('steps/')
import nnetgraph
import nnetactivations
import libs.tensorflow.kaldi_tensorflow.batchdispenser as batchdispenser
import libs.tensorflow.kaldi_tensorflow.ark as ark

## a class for a neural network that can be used together with Kaldi
class Nnet:
	#Nnet constructor
	#
	#@param conf nnet configuration
	#@param input_dim network input dimension
	#@param num_labels number of target labels
	def __init__(self, conf, input_dim, num_labels):
		
		#get nnet structure configs
		self.conf = dict(conf.items('nnet'))
		
		#define location to save neural nets
		self.conf['savedir'] = conf.get('directories','expdir') + '/' + self.conf['name']
		if not os.path.isdir(self.conf['savedir']):
			os.mkdir(self.conf['savedir'])
		if not os.path.isdir(self.conf['savedir'] + '/training'):
			os.mkdir(self.conf['savedir'] + '/training')
			
		#compute the input_dimension of the spliced features
		self.input_dim = input_dim * (2*int(self.conf['context_width']) + 1)
			
		if self.conf['batch_norm']=='True':
			activation = nnetactivations.Batchnorm_wrapper(None)
		else:
			activation = None
			
		#create the activation function
		if self.conf['nonlin'] == 'relu':
			activation = nnetactivations.Tf_wrapper(activation, tf.nn.relu)
		elif self.conf['nonlin'] == 'sigmoid':
			activation = nnetactivations.Tf_wrapper(activation, tf.nn.sigmoid)
		elif self.conf['nonlin'] == 'tanh':
			activation = nnetactivations.Tf_wrapper(activation, tf.nn.tanh)
		elif self.conf['nonlin'] == 'linear':
			activation = nnetactivations.Tf_wrapper(activation, lambda(x): x)
		else:
			raise Exception('unkown nonlinearity')
			
		if self.conf['l2_norm']=='True':
			activation = nnetactivations.L2_wrapper(activation)
			
		if float(self.conf['dropout']) < 1:
			activation = nnetactivations.Dropout_wrapper(activation, float(self.conf['dropout']))
		
			
		#create a DNN
		self.DNN = nnetgraph.DNN(num_labels, int(self.conf['num_hidden_layers']), int(self.conf['num_hidden_units']), activation, int(self.conf['add_layer_period']) > 0)
	
	## Train the neural network
	#
	#@param featdir directory where the training features are located (in feats.scp)
	#@param alifile the file containing the state alignments
	def train(self, featdir, alifile):
		
		#create a feature reader
		print("featdir: "+str(featdir))
		reader = batchdispenser.FeatureReader(featdir + '/feats_shuffled.scp', featdir + '/cmvn.scp', featdir + '/utt2spk', int(self.conf['context_width']))

		#create a batch dispenser
		dispenser = batchdispenser.Batchdispenser(reader, int(self.conf['batch_size']), alifile, self.DNN.output_dim)
	        print("###create a batch dispenser###")	
		#get the validation set
		valid_batches = [dispenser.getBatch() for _ in range(int(self.conf['valid_batches']))]
		dispenser.split()
                print("###get the validation set###")
		if len(valid_batches)>0:
			val_data = np.concatenate([val_batch[0] for val_batch in valid_batches])
			val_labels = np.concatenate([val_batch[1] for val_batch in valid_batches])
		else:
			val_data = None
			val_labels = None
		
		#compute the total number of steps
		num_steps = int(dispenser.numUtt/int(self.conf['batch_size'])*int(self.conf['num_epochs']))
					
		#set the step to the saving point that is closest to the starting step
		step = int(self.conf['starting_step']) - int(self.conf['starting_step'])%int(self.conf['check_freq'])
					
		#go to the point in the database where the training was at checkpoint
		for _ in range(step):
			dispenser.skipBatch()
		
		#create a visualisation of the DNN
		
		#put the DNN in a training environment
		trainer = nnetgraph.NnetTrainer(self.DNN, self.input_dim, float(self.conf['initial_learning_rate']), float(self.conf['learning_rate_decay']), num_steps, int(self.conf['numframes_per_batch']))
		
		#start the visualization if it is requested
		if self.conf['visualise'] == 'True':
			if os.path.isdir(self.conf['savedir'] + '/logdir'):
				shutil.rmtree(self.conf['savedir'] + '/logdir')
				
			trainer.startVisualization(self.conf['savedir'] + '/logdir')
		
		#start a tensorflow session
		config = tf.ConfigProto()
		config.gpu_options.allow_growth=True
		with tf.Session(graph=trainer.graph, config=config) as session:
			#initialise the trainer
			trainer.initialize()
			
			#load the neural net if the starting step is not 0
			if step > 0:
				trainer.restoreTrainer(self.conf['savedir'] + '/training/step' + str(step))
				
			#do a validation step
			if val_data is not None:
				validation_loss = trainer.evaluate(val_data, val_labels)
				print('validation loss at step %d: %f' %(step, validation_loss ))
				validation_step = step
				trainer.saveTrainer(self.conf['savedir'] + '/training/validated')
				num_retries = 0
			
			#start the training iteration
			while step<num_steps:
			
				#get a batch of data
				batch_data, batch_labels = dispenser.getBatch()
				
				#update the model
				loss = trainer.update(batch_data, batch_labels)
				
				#print the progress
				print('step %d/%d loss: %f' %(step, num_steps, loss))
				
				#increment the step
				step+=1
				
				#validate the model if required
				if step%int(self.conf['valid_frequency']) == 0 and val_data is not None:
					current_loss = trainer.evaluate(val_data, val_labels)
					print('validation loss at step %d: %f' %(step, current_loss))
					
					if self.conf['valid_adapt'] == 'True':
						#if the loss increased, half the learning rate and go back to the previous validation step
						if current_loss > validation_loss:
						
							#go back in the dispenser
							for _ in range(step-validation_step):
								dispenser.returnBatch()
							
							#load the validated model
							trainer.restoreTrainer(self.conf['savedir'] + '/training/validated')
							trainer.halve_learning_rate()
							step = validation_step
							
							if num_retries == int(self.conf['valid_retries']):
								print('the validation loss is worse, terminating training')
								break
								
							print('the validation loss is worse, returning to the previously validated model with halved learning rate')
							
							num_retries+=1
							
							continue
						
						else:
							validation_loss=current_loss
							validation_step = step
							num_retries=0
							trainer.saveTrainer(self.conf['savedir'] + '/training/validated')
							
				#add a layer if its required
				if int(self.conf['add_layer_period']) > 0:
					if step%int(self.conf['add_layer_period']) == 0 and step/int(self.conf['add_layer_period']) < int(self.conf['num_hidden_layers']):
					
						print('adding layer, the model now holds %d/%d layers' %(step/int(self.conf['add_layer_period']) + 1, int(self.conf['num_hidden_layers'])))
						trainer.control_ops['add'].run()
						trainer.control_ops['init'].run()
					
						#do a validation step
						validation_loss = trainer.evaluate(val_data, val_labels)
						print('validation loss at step %d: %f' %(step, validation_loss ))
						validation_step = step
						trainer.saveTrainer(self.conf['savedir'] + '/training/validated')
						num_retries = 0
							
				#save the model if at checkpoint
				if step%int(self.conf['check_freq']) == 0:
					trainer.saveTrainer(self.conf['savedir'] + '/training/step' + str(step))
					
						
			#compute the state prior and write it to the savedir
			prior = dispenser.computePrior()
			np.save(self.conf['savedir'] + '/prior.npy', prior)
						
			#save the final model
			trainer.saveModel(self.conf['savedir'] + '/final')
			
				
				

	##compute pseudo likelihoods the testing set
	#
	#@param featdir directory where the features are located (in feats.scp)
	#@param decodir location where output will be stored (in feats.scp), cannot be the same as featdir
	def decode(self, featdir, decodedir):
		#create a feature reader
		reader = batchdispenser.FeatureReader(featdir + '/feats.scp', featdir + '/cmvn.scp', featdir + '/utt2spk', int(self.conf['context_width']))
	
		#remove ark file if it allready exists
		if os.path.isfile(decodedir + '/feats.ark'):
			os.remove(decodedir + '/feats.ark')
		
		#open likelihood writer
		writer = ark.ArkWriter(decodedir + '/feats.scp')
		
		#create a decoder
		decoder = nnetgraph.NnetDecoder(self.DNN, self.input_dim)
		
		#read the prior
		prior = np.load(self.conf['savedir'] + '/prior.npy')
	
		#start tensorflow session
		config = tf.ConfigProto()
		config.gpu_options.allow_growth=True
		with tf.Session(graph=decoder.graph, config=config) as session:
		
			#load the model
			decoder.restore(self.conf['savedir'] + '/final')
	
			#feed the utterances one by one to the neural net
			while True:
				utt_id, utt_mat, looped = reader.getUtt()
			
				if looped:
					break
			
				#compute predictions
				output = decoder(utt_mat)
				
				#get state likelihoods by dividing by the prior
				output = output/prior
				
				#floor the values to avoid problems with log
				np.where(output == 0,np.finfo(float).eps,output)

				#write the pseudo-likelihoods in kaldi feature format
				writer.write_next_utt(decodedir + '/feats.ark', utt_id, np.log(output))
		
		#close the writer
		writer.close()
