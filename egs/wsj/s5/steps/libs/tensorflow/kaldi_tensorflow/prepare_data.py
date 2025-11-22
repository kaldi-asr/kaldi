##@package prepare_data
# contains the functions used to prepare the data for GMM and DNN training 

import numpy as np
import gzip
import scipy.io.wavfile as wav
import os
from shutil import copyfile
from random import shuffle

import ark
import feat
import kaldiInterface

##This function will compute the features of all segments and save them on disk
#
#@param datadir directory where the kaldi data prep has been done
#@param featdir directory where the features will be put
#@param conf feature configuration
#@param feat_type type of features to be computed, options are mfcc, fbank and ssc
#@param dynamic the type of dynamic information added, options are nodelta, delta and ddelta
def prepare_data(datadir, featdir, conf, feat_type, dynamic):
	
	if not os.path.exists(featdir):
		os.makedirs(featdir)
	
	#read the segments
	if os.path.isfile(datadir + '/segments'):
		segments = kaldiInterface.read_segments(datadir + '/segments')
		found_segments = True
	else:
		print('WARNING: no segments file found, assuming each wav file is seperate utterance')
		found_segments = False

	#read the wavfiles
	wavfiles = kaldiInterface.read_wavfiles(datadir + '/wav.scp')
	
	#create ark writer
	writer = ark.ArkWriter(featdir + '/feats.scp')
	if os.path.isfile(featdir + '/feats.ark'):
		os.remove(featdir + '/feats.ark')
		
	#read all the wav files
	RateUtt = {utt: read_wav(wavfiles[utt]) for utt in wavfiles}
		
	#create a featureComputer
	comp = feat.FeatureComputer(feat_type, dynamic, conf)
	
	#compute all the features
	for utt in wavfiles:
		if found_segments:
			for seg in segments[utt]:
				features = comp(RateUtt[utt][1][int(seg[1]*RateUtt[utt][0]):int(seg[2]*RateUtt[utt][0])], RateUtt[utt][0])
				writer.write_next_utt(featdir + '/feats.ark', seg[0], features)
		else:
			features = comp(RateUtt[utt][1], RateUtt[utt][0])
			writer.write_next_utt(featdir + '/feats.ark', utt, features)

	writer.close()
	
	#copy some kaldi files to features dir
	copyfile(datadir + '/utt2spk', featdir + '/utt2spk')
	copyfile(datadir + '/spk2utt', featdir + '/spk2utt')
	copyfile(datadir + '/text', featdir + '/text')
	copyfile(datadir + '/wav.scp', featdir + '/wav.scp')
	
## compute the cmvn statistics and save them
#
#@param featdir the directory containing the features in feats.scp
def compute_cmvn(featdir):
	#read the spk2utt file
	spk2utt = open(featdir + '/spk2utt', 'r')
	
	#create feature reader
	reader = ark.ArkReader(featdir + '/feats.scp')
	
	#create writer for cmvn stats
	writer = ark.ArkWriter(featdir + '/cmvn.scp')
	
	#loop over speakers
	for line in spk2utt:
		#cut off end of line character
		line = line[0:len(line)-1] 
	
		split = line.split(' ')
		
		#get first speaker utterance
		spk_data = reader.read_utt(split[1])
		
		#get the rest of the utterances
		for utt_id in split[2:len(split)]:
			spk_data = np.append(spk_data, reader.read_utt(utt_id), axis=0)
			
		#compute mean and variance
		stats = np.zeros([2,spk_data.shape[1]+1])
		stats[0,0:spk_data.shape[1]] = np.sum(spk_data, 0)
		stats[1,0:spk_data.shape[1]] = np.sum(np.square(spk_data),0)
		stats[0, spk_data.shape[1]] = spk_data.shape[0]
		
		#write stats to file
		writer.write_next_utt(featdir + '/cmvn.ark', split[0], stats)
	
	writer.close()
	
## shuffle the utterances and put them in feats_shuffled.scp
#
#@param featdir the directory containing the features in feats.scp
def shuffle_examples(featdir):
	#read feats.scp
	featsfile = open(featdir + '/feats.scp', 'r')
	feats = featsfile.readlines()
	
	#shuffle feats randomly
	shuffle(feats)

	#wite them to feats_shuffled.scp
	feats_shuffledfile = open(featdir + '/feats_shuffled.scp', 'w')
	feats_shuffledfile.writelines(feats)

## read a wav file formatted by kaldi
#
#@param wavfile a pair containing eiher the filaname or the command to read the wavfile and a boolean that determines if its a name or a command
def read_wav(wavfile):
	if wavfile[1]:
		#read the audio file and temporarily copy it to tmp (and duplicate, I don't know how to avoid this)
		os.system(wavfile[0] + ' tee tmp.wav > duplicate.wav')
		#read the created wav file
		(rate,utterance) = wav.read('tmp.wav')
		#delete the create file
		os.remove('tmp.wav')
		os.remove('duplicate.wav')
	else:
		(rate,utterance) = wav.read(wavfile[0])
		
	return rate, utterance
	
		
	
		
	
	

