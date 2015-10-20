#!/usr/bin/env python
import sys, os, logging, numpy as np
#os.environ['THEANO_FLAGS']='nvcc.flags=-arch=sm_30,mode=FAST_RUN,device=gpu,floatX=float32'
os.environ['THEANO_FLAGS']='device=gpu,floatX=float32'
import theano, theano.tensor as T

import scipy.io.wavfile
import itertools, shutil
import cPickle as pickle
from collections import OrderedDict

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import parse_config
from io_funcs import kaldi_io, utils
from feature_funcs import features, feature_preprocess

from neuralnet import NeuralNet
import neuralnet
from feature_funcs.feature_preprocess import FeaturePreprocess, CMVN


import mkl
mkl.set_num_threads(1)
import numexpr
numexpr.set_num_threads(1)

np.random.seed(42)
from scipy.stats import bernoulli
##################################
def apply_mask_to_minibatch(minibatch, strm_indices, pvals=None):

  nstrms=len(strm_indices)
  ncombs=np.power(2, nstrms)-1
  if pvals==None:
    pvals=[1.0/ncombs]*ncombs

  for X, t in minibatch:
    assert strm_indices[-1][-1] == X.shape[1]
    
    wts = np.zeros((X.shape[0], nstrms))
    #get weights
    comb_idxs=np.random.multinomial(1, pvals, size=X.shape[0])
    for ii, c in enumerate(comb_idxs):
      bin_str='{:010b}'.format(np.nonzero(c)[0][0]+1)
      bin_str=bin_str[-nstrms:]
      wts[ii,:] = np.asarray(map(lambda x: int(x), bin_str))

    #apply weights
    for ii, (st, en) in enumerate(strm_indices):
      X[:, st:en] = X[:, st:en] * wts[:, [ii]]

    yield X, t
    
##################################

from optparse import OptionParser
usage = "%prog [options] <trn-scp> <trn-labels> <outdir>"
parser = OptionParser(usage)

parser.add_option('--config', dest="config",
                  help="Configuration file to read (this option may be repeated) [default: %default]",
                  default="", type='string')

parser.add_option('--strm-indices', dest="strm_indices",
                  help="Indices for application of masks [default: %default]",
                  default="", type="string")

parser = parse_config.theano_nnet_parse_opts(parser)

(o, args) = parser.parse_args()
# options specified in config overides command line
if o.config != "": (o, args) = parse_config.parse_config(parser, o.config)

if len(args) != 3:
  parser.print_help()
  sys.exit(1)

(trn_scp, trn_lab, outdir) = (args[0], args[1], args[2])
utils.mkdir_p(outdir)
utils.mkdir_p(outdir+"/nnet")

## Create log file
logging.basicConfig(filename=outdir+'/train.log', format='%(asctime)s: %(message)s', level=logging.INFO)
logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))

logging.info("Running as %s ", " ".join(sys.argv))

if len(o.strm_indices) == 0:
  logging.error("Use train.py for normal DNN training")
  sys.exit(1)
a = map(lambda x: int(x), o.strm_indices.split(","))
o.strm_indices = []
for st, en in zip(a[0:-1], a[1:]):
  o.strm_indices.append((st, en))

## Check final_nnet.pkl exist, if yes quit
if os.path.exists(outdir+"/final_nnet.pklz"):
  logging.info("Skipping training as %s/final_nnet.pklz exists", outdir)
  sys.exit(0)

## Load labels ##
logging.info("Loading train labels from %s", trn_lab)
trn_lab_dict = utils.labels_ascii_to_dict(trn_lab)
##

##
feat_preprocess = FeaturePreprocess(o)

## Load cmvn ##
trn_cmvn = CMVN(feat_preprocess, o.trn_utt2spk_file, o.trn_cmvn_scp)
##

## Check if to start from initial or middle
last_iter, last_learn_rate, nSamples, best_nnet_file = utils.theano_nnet_start_ExpLrSched(outdir, o)
if last_iter != 0:
  logging.info("last_iter=%d, last_learn_rate=%f, nSamples=%d, best_nnet_file=%s", last_iter, last_learn_rate, nSamples, best_nnet_file)
  # creating theano neural network
  X_ = T.matrix("X")
  nnet = NeuralNet()
  nnet.Load(X_, best_nnet_file)

  #load feat_preprocess.pkl, it should exist
  if os.path.exists(outdir+"/feat_preprocess.pkl") == False:
    logging.info("Expects %s/feat_preprocess.pkl . Something might be wrong, better to restart from begining", outdir);
    sys.exit(1)
  logging.info("Loading feat preprocess from %s/feat_preprocess.pkl", outdir)
  feat_preprocess = pickle.load(open(outdir+"/feat_preprocess.pkl", "rb"))

else:
  ## input mean and variance normalization
  logging.info("Estimating mean and std at the NN input")
  N = F = S = 0.0
  with kaldi_io.KaldiScpReader(trn_scp, feature_preprocess.full_preprocess, reader_args=[feat_preprocess, trn_cmvn.utt2spk_dict, trn_cmvn.cmvn_dict]) as trn_data_it:
    for ii, (X, utt) in enumerate(trn_data_it):
      if ii == 10000: #10k utterances
        trn_data_it.close()
        break
      N += len(X)
      F += np.sum(X, axis=0)
      S += np.sum(X**2, axis=0)
  input_mean = (F/N).astype(T.config.floatX)
  input_std  = np.sqrt(S/N - input_mean**2).astype(T.config.floatX)

  #put it in feat_preprocess
  (feat_preprocess.glob_mean, feat_preprocess.glob_std) = (input_mean, input_std)

  # creating theano neural network
  logging.info("Creating and initializing NN")
  nnet = NeuralNet()
  nnet.initialize_from_proto(o.nnet_proto)
  #nnet.set_name('dnn')
  
  # save feat_preprocess.pkl
  logging.info("Saving feat preprocess to %s/feat_preprocess.pkl", outdir)
  # save feature_preprocess, for further usage 
  pickle.dump(feat_preprocess, open(outdir+"/feat_preprocess.pkl", "wb"))

  # save initial net
  logging.info("Saving initial nnet to %s/nnet_initial.pklz", outdir)
  nnet.save_weights(outdir+"/nnet_initial.pklz")
  nnet_best_file = outdir+"/nnet_initial.pklz"

logging.info("Creating theano functions")
(X_, Y_) = (nnet.X, nnet.Y)
lr_ = T.scalar()
T_ = T.ivector("T")
cost_ = T.nnet.categorical_crossentropy(Y_, T_).sum()
acc_ = T.eq(T.argmax(Y_, axis=1), T_).sum()

params = nnet.params
lr_coefs = nnet.lr_coefs
gparams = T.grad(cost_, params)

# compute list of weights updatas
updates = OrderedDict()
for param, gparam, lr_coef in zip(params, gparams, lr_coefs):
  updates[param] = param - (gparam * lr_ * lr_coef)

train = theano.function(
  inputs=[X_, T_, lr_],
  outputs=[cost_, acc_],
  updates=updates)

xentropy = theano.function(inputs=[X_, T_], outputs=[cost_, acc_])

(start_iter, max_iters) = (last_iter+1, o.max_iters)
(segment_buffer_size, batch_size) = (o.segment_buffer_size, o.batch_size) 
(start_learn_rate, end_learn_rate) = (o.start_learn_rate, o.end_learn_rate)
(decay_factor, keep_lr_iters) = (o.decay_factor, o.keep_lr_iters)
strm_indices = o.strm_indices

#exp_lr based on 
#An Empirical study of learning rates in deep neural networks for speech recognition, Senior et. al.

# start training
lr = start_learn_rate
for kk in range(start_iter, max_iters):
  np.random.seed(42)
  logging.info("Training epoch: %02d, learning rate: %f", kk, lr)
  error = accuracy = n = 0.0
  with kaldi_io.KaldiScpReader(trn_scp, feature_preprocess.full_preprocess, reader_args=[feat_preprocess, trn_cmvn.utt2spk_dict, trn_cmvn.cmvn_dict]) as trn_data_it:
    for mm, segment_buffer in enumerate(utils.isplit_every(features.fea_lab_pair_iter(trn_data_it, trn_lab_dict), segment_buffer_size)):
      noncum_error = noncum_accuracy = noncum_n = 0.0
      for X, t in apply_mask_to_minibatch(utils.segment_buffer_to_minibatch_iter(segment_buffer, batch_size), strm_indices):
        if kk > keep_lr_iters:
          nSamples += len(X)

        lr = start_learn_rate * np.power(10, -nSamples/decay_factor)
        lr = lr.astype('float32')

        err, acc = train(X, t, lr)
        error += err; accuracy += acc; n += len(X); 
        noncum_error += err; noncum_accuracy += acc; noncum_n += len(X)

      logging.info("%f | %f | %f | %f | %d | %f", error / n, accuracy /n, noncum_error / noncum_n, noncum_accuracy / noncum_n, nSamples, lr)
  (trn_error, trn_accuracy) = (error / n, accuracy / n)

  this_nnet_file_tmp_name = "%s/nnet/nnet_epoch%02d.pklz" %(outdir, kk)
  nnet.save_weights(this_nnet_file_tmp_name)

  # change the above nnet 
  this_nnet_file = "%s/nnet/nnet_epoch%02d_learnrate%f_tr%f.pklz" %(outdir, kk, lr, trn_error)
  logging.info("Saving epoch%d nnet to %s", kk, this_nnet_file)
  shutil.move(this_nnet_file_tmp_name, this_nnet_file)

  done_fd = open("%s/.done_epoch%d" %(outdir, kk), "w")
  done_fd.write("--train-error=%f\n--train-accuracy=%f\n--learn-rate=%f\n--nSamples=%d\n" %(trn_error, trn_accuracy, lr, nSamples))
  done_fd.close()

  best_nnet_file = this_nnet_file
  best_nnet_fd = open("%s/.best_nnet" %(outdir), "w")
  best_nnet_fd.write("%s" %(best_nnet_file))
  best_nnet_fd.close()

  if lr <= end_learn_rate:
    break

# softlink to final_nnet.pkl
old_dir = os.path.abspath(os.getcwd())
os.chdir(outdir)
rel_best_file = "/".join(best_nnet_file.split("/")[-2:])
soft_link_cmd="ln -s %s final_nnet.pklz" %(rel_best_file)
os.system(soft_link_cmd)
os.chdir(old_dir)

if os.path.exists(outdir+"/final_nnet.pklz"):
  logging.info("Succeeded training the Neural Network : %s", outdir+"/final_nnet.pklz")
else:
  logging.info("Error training neural network...")
  sys.exit(1)

sys.exit(0)

