# Copyright 2016    Vincent Renkens
#           2016    Gaofeng Cheng
###### Attention please######
## tensorflow version: 0.11.0rc2
## python version: python2.7
## only supporting: 
##                reading features(compress = false) made by kaldi; training neural network with tensorflow.
## This version now only support DNN running on one gpu card, further functions need to be implemented.
## before running this ,you should 'source path.sh' first
## if you want to run tensor-kaldi on another dataset, you need to do some work
## Results(without ivector)         epoch   eval92_bd   eval92 
## kaldi 2048 relu 6 layer          8       4.22        6.59
## tensorflow + kaldi(see config)   ~3      4.93        6.95
##
from six.moves import configparser
import os
import sys

sys.path.append('steps/')
sys.path.insert(0, os.path.realpath(os.path.dirname(sys.argv[0])) + '/')

import libs.tensorflow.tensorflow_nnet.nnet as nnet
import libs.tensorflow.kaldi_tensorflow.ark as ark
import libs.tensorflow.kaldi_tensorflow.kaldiInterface as kaldiInterface
import libs.tensorflow.kaldi_tensorflow.batchdispenser as batchdispenser

TRAIN_NNET = True			#required
TEST_NNET = True			#required if the performance of the DNN is tested

# read config file
config = configparser.ConfigParser()
config.read('conf/wsj_tensorflow_kaldi.conf')
current_dir = os.getcwd()

# setting setting gpu used
os.environ['CUDA_VISIBLE_DEVICES']=config.get('general','gpu_card_id')

# get the feature input dim
reader = ark.ArkReader(config.get('directories','train_features') + '/' + config.get('dnn-features','name') + '/feats.scp')
(_,features,_) = reader.read_next_utt()
input_dim = features.shape[1]

# get number of output labels
graph_dir_split = config.get('nnet','graph_dir').strip().split(',')
if len(graph_dir_split) is 0:
	print("Error: there should be lang dir")
	exit()

# here we suppose the pdf num shared by different graph_dir is the same
gmm_dir = config.get('directories','expdir') + '/' + config.get('nnet','gmm_name')
gmm_ali_dir = config.get('directories','expdir') + '/' + config.get('nnet','gmm_ali_name')
numpdfs = open(gmm_dir + '/'+graph_dir_split[0]+'/num_pdfs')
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
		kaldiInterface.shuffle_examples(config.get('directories','train_features') + '/' +  config.get('dnn-features','name'))
		print("train_data_root: "+str(config.get('directories','train_features') + '/' +  config.get('dnn-features','name')))
	#put all the alignments in one file
	gmm_dir_ali = gmm_ali_dir+'/ali.*.gz'
	os.system('ls %s | grep [0-9] | wc -l > %s' % (gmm_dir_ali,config.get('general','gmm_ali_number')))
	f = open(config.get('general','gmm_ali_number'), 'r')
	gmm_ali_jobs_num = f.read()
	f.close()
	if int(gmm_ali_jobs_num) is 0:
		print("Error: gmm ali reading seems going wrong")
		exit()
	alifiles = [gmm_ali_dir + '/ali.' + str(i+1) + '.gz' for i in range(int(gmm_ali_jobs_num))]
	alifile = gmm_ali_dir + '/ali.all.gz'
	alifile_unzip = gmm_ali_dir + '/ali.all'
	ali_final_mdl = config.get('directories','expdir') + '/'+ config.get('nnet','gmm_ali_name') + '/final.mdl'
	alifile_in_pdftxt = gmm_ali_dir + '/ali.all.pdf.txt'
	alifile_in_pdftxt_gzipped = gmm_ali_dir + '/ali.all.pdf.txt.gz'

	print('alifile: ' + alifile)
	print('alifile_unzip: ' + alifile_unzip)
	print('ali_final_mdl: ' + ali_final_mdl)
	print('alifile_in_pdftxt: ' + alifile_in_pdftxt)
	os.system('source path.sh')
	os.system('cat %s > %s' % (' '.join(alifiles), alifile))
	os.system('gunzip -c %s > %s' % (alifile, alifile_unzip))
	os.system('ali-to-pdf %s ark:%s ark,t:%s' % (ali_final_mdl, alifile_unzip, alifile_in_pdftxt)) 
	os.system('gzip -c %s > %s' % (alifile_in_pdftxt, alifile_in_pdftxt_gzipped))    	
	#train the neural net
	print('------- training neural net ----------')
	Nnet.train(config.get('directories','train_features') + '/' +  config.get('dnn-features','name'), alifile_in_pdftxt_gzipped)

if TEST_NNET:

	#use the neural net to calculate posteriors for the testing set
	print('------- computing state pseudo-likelihoods ----------')
	savedir = config.get('directories','expdir') + '/' + config.get('nnet','name')
	print("pseudo-likelihoods savedir: "+savedir)

	# decoding test feature with different graph_dir, of course, test feature dir always more than 1 
	test_feature_dir = config.get('dnn-features','test_feature_name')
	test_feature_dir_split = test_feature_dir.strip().split(',')
    
	# decoding all test feature sets and lang dir
	for graph_dir_x in graph_dir_split:
		for test_feature_dir_x in test_feature_dir_split:
			decodedir = savedir + '/decode'+ graph_dir_x + test_feature_dir_x
			print("decodedir: "+decodedir)
			if not os.path.isdir(decodedir):
				os.mkdir(decodedir)

			Nnet.decode('data/'+test_feature_dir_x , decodedir)
			print('------- decoding testing sets ----------')
			#copy the gmm model and some files to speaker mapping to the decoding dir
			os.system('cp %s %s' %(gmm_ali_dir + '/final.mdl', decodedir))
			os.system('cp -r %s %s' %(gmm_dir + '/' + graph_dir_x, decodedir))
			os.system('cp %s %s' %(config.get('directories','test_features') + '/' + test_feature_dir_x + '/utt2spk', decodedir))
			os.system('cp %s %s' %(config.get('directories','test_features') + '/' + test_feature_dir_x + '/text', decodedir))
			num_job=8
			#decode using kaldi
			os.system('./steps/decode_tensorflow.sh --cmd %s --nj %s %s/graph* %s %s/kaldi_decode | tee %s/decode.log || exit 1;' % ( config.get('general','cmd'), num_job, decodedir, decodedir, decodedir, decodedir))
			
			#get results
			os.system('grep WER %s/kaldi_decode/wer_* | utils/best_wer.sh' % decodedir)

	
	
	
	
	
	

