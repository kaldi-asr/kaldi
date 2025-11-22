# Copyright 2016    Vincent Renkens
##@package batchdispenser
# contain the functionality for read features and batches of features for neural network training and testing

import numpy as np

import ark
import kaldiInterface

## Class that can read features from a Kaldi archive and process them (cmvn and splicing)
class FeatureReader:
	##create a FeatureReader object
	#
	#@param scpfile: path to the features .scp file
	#@param cmvnfile: path to the cmvn file
	#@param utt2spkfile:path to the file containing the mapping from utterance ID to speaker ID
	#@param context_width: context width for splicing the features
	def __init__(self, scpfile, cmvnfile, utt2spkfile, context_width):
		#create the feature reader
		self.reader = ark.ArkReader(scpfile)
		
		#create a reader for the cmvn statistics
		self.reader_cmvn = ark.ArkReader(cmvnfile)
		
		#save the utterance to speaker mapping
		self.utt2spk = kaldiInterface.read_utt2spk(utt2spkfile)
		
		#store the context width
		self.context_width = context_width
		
	##read the next features from the archive, normalize and splice them
	#
	#@return the normalized and spliced features 
	def getUtt(self):
		#read utterance
		(utt_id, utt_mat, looped) = self.reader.read_next_utt()
		
		#apply cmvn
		cmvn_stats = self.reader_cmvn.read_utt(self.utt2spk[utt_id])
		utt_mat = apply_cmvn(utt_mat, cmvn_stats)
		
		#splice the utterance
		utt_mat = splice(utt_mat,self.context_width)
		
		return utt_id, utt_mat, looped
	
	##only gets the ID of the next utterance (also moves forward in the reader)
	#
	#@return the ID of the uterance
	def nextId(self):
		return self.reader.read_next_scp()
		
	##only gets the ID of the previous utterance (also moves backward in the reader)
	#
	#@return the ID of the uterance
	def prevId(self):
		return self.reader.read_previous_scp()
		
	##split of the features that have been read so far
	def split(self):
		self.reader.split()
		
## Class that dispenses batches of data for mini-batch training
class Batchdispenser:
	##Batchdispenser constructor
	#
	#@param featureReader: a feature reader object
	#@param size: the batch size
	#@param scpfile: the path to the features .scp file
	#@param alifile: the path to the file containing the alignments
	#@param num_labels: total number of labels
	def __init__(self, featureReader, size, alifile, num_labels):
		
		#store the feature reader
		self.featureReader = featureReader
		print("alifile: "+str(alifile))
		#read the alignments
		self.alignments = kaldiInterface.read_alignments(alifile)
		#save the number of labels
		self.num_labels = num_labels
		
		#store the batch size
		self.size = size
	
	##get a batch of features and alignments in one-hot encoding 
	#
	#@return a batch of data, the corresponding labels in one hot encoding
	def getBatch(self):
		
		n=0
		batch_data = np.empty(0)
		batch_labels = np.empty(0)
		while n < self.size:
			#read utterance
			utt_id, utt_mat, _ = self.featureReader.getUtt()
			
			#check if utterance has an alignment
			if utt_id in self.alignments:

				#add the features and alignments to the batch
				batch_data = np.append(batch_data, utt_mat)
				batch_labels = np.append(batch_labels, self.alignments[utt_id])
					
				#update number of utterances in the batch
				n += 1
			else:
				print('WARNING no alignment for %s' % utt_id)
		
		#reahape the batch data
		batch_data = batch_data.reshape(batch_data.size/utt_mat.shape[1], utt_mat.shape[1])
		
		#put labels in one hot encoding
		batch_labels = (np.arange(self.num_labels) == batch_labels[:,np.newaxis]).astype(np.float32)
		
		return (batch_data, batch_labels)
	
	##split of the part that has allready been read by the batchdispenser, this can be used to read a validation set and then split it of from the rest
	def split(self):
		self.featureReader.split()
	
	##skip a batch
	def skipBatch(self):
		n=0
		while n < self.size:
			#read utterance
			utt_id = self.featureReader.nextId()
			
			#check if utterance has an alignment
			if utt_id in self.alignments:
					
				#update number of utterances in the batch
				n += 1
	
	##return to the previous batch			
	def returnBatch(self):
		n=0
		while n < self.size:
			#read utterance
			utt_id = self.featureReader.prevId()
			
			#check if utterance has an alignment
			if utt_id in self.alignments:
					
				#update number of utterances in the batch
				n += 1
		
	##compute the pior probability of the labels in alignments
	#
	#@return a numpy array containing the label prior probabilities
	def computePrior(self):
		prior = np.array([(np.arange(self.num_labels) == alignment[:,np.newaxis]).astype(np.float32).sum(0) for alignment in self.alignments.values()]).sum(0)
		return prior/prior.sum()
		
	##the number of utterances
	@property
	def numUtt(self):
		return len(self.alignments)
		
	
	
##apply mean and variance normalisation based on the previously computed statistics
#
#@param utt the utterance feature numpy matrix
#@param stats a numpy array containing the mean and variance statistics. The first row contains the sum of all the fautures and as a last element the total numbe of features. The second row contains the squared sum of the features and a zero at the end
#
#@return a numpy array containing the mean and variance normalized features
def apply_cmvn(utt, stats):
	
	#compute mean
	mean = stats[0,:-1]/stats[0,-1]
	
	#compute variance
	variance = stats[1,:-1]/stats[0,-1] - np.square(mean)
	
	#return mean and variance normalised utterance
	return np.divide(np.subtract(utt, mean), np.sqrt(variance))
	
##splice the utterance
#
#@param utt numpy matrix containing the utterance features to be spliced
#@param context width how many frames to the left and right should be concatenated
#
#@return a numpy array containing the spliced features
def splice(utt, context_width):
	
	#create spliced utterance holder
	utt_spliced = np.zeros(shape = [utt.shape[0],utt.shape[1]*(1+2*context_width)], dtype=np.float32)
	
	#middle part is just the uttarnce
	utt_spliced[:,context_width*utt.shape[1]:(context_width+1)*utt.shape[1]] = utt
	
	for i in range(context_width):
		
		#add left context
		utt_spliced[i+1:utt_spliced.shape[0], (context_width-i-1)*utt.shape[1]:(context_width-i)*utt.shape[1]] = utt[0:utt.shape[0]-i-1,:]
	 	
	 	#add right context	
		utt_spliced[0:utt_spliced.shape[0]-i-1, (context_width+i+1)*utt.shape[1]:(context_width+i+2)*utt.shape[1]] = utt[i+1:utt.shape[0],:]
	
	return utt_spliced
