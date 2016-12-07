# Copyright 2016    Vincent Renkens
#           2016    Gaofeng Cheng

# reading features(compress = false) made by kaldi, training neural network with tensorflow
from six.moves import configparser
import os
import sys

sys.path.append('steps/')
sys.path.append('features')
sys.path.append('io')
sys.path.append('kaldi')
sys.path.append('neuralNetworks')

import nnet
import ark
import kaldiInterface
import prepare_data

TRAIN_NNET = True			#required
TEST_NNET = True			#required if the performance of the DNN is tested

#read config file
config = configparser.ConfigParser()
config.read('config/config_AURORA4.cfg')
current_dir = os.getcwd()

#get the feature input dim
reader = ark.ArkReader(config.get('directories','train_features') + '/' + config.get('dnn-features','name') + '/feats.scp')
(_,features,_) = reader.read_next_utt()
input_dim = features.shape[1]

#get number of output labels
numpdfs = open(config.get('directories','expdir') + '/' + config.get('nnet','gmm_name') + '/graph_tgpr/num_pdfs')
num_labels = numpdfs.read()
num_labels = int(num_labels[0:len(num_labels)-1])
numpdfs.close()
print("num_labels: "+str(num_labels))	
print("input_dim: "+str(input_dim))

#create the neural net 	
Nnet = nnet.Nnet(config, input_dim, num_labels)

if TRAIN_NNET:

	#only shuffle if we start with initialisation
	if config.get('nnet','starting_step') == '0':
		#shuffle the examples on disk
		print('------- shuffling examples ----------')
		prepare_data.shuffle_examples(config.get('directories','train_features') + '/' +  config.get('dnn-features','name'))
		print("train_data_root: "+str(config.get('directories','train_features') + '/' +  config.get('dnn-features','name')))
	#put all the alignments in one file
	alifiles = [config.get('directories','expdir') + '/' + config.get('nnet','gmm_name') + '/ali.' + str(i+1) + '.gz' for i in range(int(config.get('general','num_jobs')))]
	alifile = config.get('directories','expdir') + '/' + config.get('nnet','gmm_name') + '/ali.all.gz'
	alifile_unzip = config.get('directories','expdir') + '/' + config.get('nnet','gmm_name') + '/ali.all'
	ali_final_mdl = config.get('directories','expdir') + '/'+ config.get('nnet','gmm_name') + '/final.mdl'
	alifile_in_pdftxt = config.get('directories','expdir') + '/' + config.get('nnet','gmm_name') + '/ali.all.pdf.txt'
	alifile_in_pdftxt_gzipped = config.get('directories','expdir') + '/' + config.get('nnet','gmm_name') + '/ali.all.pdf.txt.gz'
	print('alifile: '+alifile)
	print('alifile_unzip: '+alifile_unzip)
	print('ali_final_mdl: '+ali_final_mdl)
	print('alifile_in_pdftxt: '+alifile_in_pdftxt)
	os.system('cat %s > %s' % (' '.join(alifiles), alifile))
	os.system('gunzip -c %s > %s' % (alifile, alifile_unzip))
	os.system('ali-to-pdf %s ark:%s ark,t:%s' % (ali_final_mdl, alifile_unzip, alifile_in_pdftxt)) 
	os.system('gzip -c %s > %s' % (alifile_in_pdftxt, alifile_in_pdftxt_gzipped))    	
	#train the neural net

	print('------- training neural net ----------')
	Nnet.train(config.get('directories','train_features') + '/' +  config.get('dnn-features','name'), alifile_in_pdftxt_gzipped)
exit()
if TEST_NNET:

	#use the neural net to calculate posteriors for the testing set
	print('------- computing state pseudo-likelihoods ----------')
	savedir = config.get('directories','expdir') + '/' + config.get('nnet','name')
	print("pseudo-likelihoods savedir: "+savedir)
	decodedir = savedir + '/decode'
	print("decodedir: "+decodedir)
	if not os.path.isdir(decodedir):
		os.mkdir(decodedir)
	Nnet.decode(config.get('directories','test_features') + '/' +  config.get('dnn-features','test_feature_name'), decodedir)

	print('------- decoding testing sets ----------')
	#copy the gmm model and some files to speaker mapping to the decoding dir
	os.system('cp %s %s' %(config.get('directories','expdir') + '/' + config.get('nnet','gmm_name') + '/final.mdl', decodedir))
	os.system('cp -r %s %s' %(config.get('directories','expdir') + '/' + config.get('nnet','gmm_name') + '/graph_tgpr', decodedir))
	os.system('cp %s %s' %(config.get('directories','test_features') + '/' +  config.get('dnn-features','test_feature_name') + '/utt2spk', decodedir))
	os.system('cp %s %s' %(config.get('directories','test_features') + '/' +  config.get('dnn-features','test_feature_name') + '/text', decodedir))
		
	#change directory to kaldi egs
	os.chdir(config.get('directories','kaldi_egs'))
	
	#decode using kaldi
	os.system('%s/kaldi/decode.sh --cmd %s --nj %s %s/graph %s %s/kaldi_decode | tee %s/decode.log || exit 1;' % (current_dir, config.get('general','cmd'), config.get('general','num_jobs'), decodedir, decodedir, decodedir, decodedir))
	
	#get results
	os.system('grep WER %s/kaldi_decode/wer_* | utils/best_wer.sh' % decodedir)
	
	#go back to working dir
	os.chdir(current_dir)
	
	
	
	
	

