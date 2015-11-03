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
usage = "%prog [options] <trn-scp> <trn-labels> <nnet-in> <nnet-out>"
parser = OptionParser(usage)

parser.add_option('--config', dest="config",
                  help="Configuration file to read (this option may be repeated) [default: %default]",
                  default="", type='string')

parser.add_option('--feat-preprocess', dest="feat_preprocess_file",
                  help="Feature preprocess file [default: %default]",
                  default="", type='string')

parser.add_option('--strm-indices', dest="strm_indices",
                  help="Indices for application of masks [default: %default]",
                  default="", type="string")
parser.add_option('--iters-per-epoch', dest="iters_per_epoch",
                  help="Iterations per epoch [default: %default]",
                  default=1, type=int)

parser.add_option('--done-file', dest="done_file",
                  help="Done file to dump [default: %default]",
                  default="", type='string')

parser = parse_config.theano_nnet_parse_opts(parser)

(o, args) = parser.parse_args()
# options specified in config overides command line
if o.config != "": (o, args) = parse_config.parse_config(parser, o.config)

if len(args) != 4:
  parser.print_help()
  sys.exit(1)

(trn_scp, trn_lab, nnet_in_file, nnet_out_file) = (args[0], args[1], args[2], args[3])

## Create log file
logging.basicConfig(stream=sys.stdout, format='%(asctime)s: %(message)s', level=logging.INFO)

logging.info(" Running as %s ", sys.argv[0])
logging.info(" %s", " ".join(sys.argv))

if len(o.strm_indices) == 0:
  logging.error("Use train_1iter.py for normal DNN training")
  sys.exit(1)
a = map(lambda x: int(x), o.strm_indices.split(","))
o.strm_indices = []
for st, en in zip(a[0:-1], a[1:]):
  o.strm_indices.append((st, en))

## Load labels ##
logging.info(" Loading train labels from %s", trn_lab)
trn_lab_dict = utils.labels_ascii_to_dict(trn_lab)
##

## feature preprocessing ##
if o.feat_preprocess_file != "":
  logging.info(" Loading feature preprocess from %s", o.feat_preprocess_file)
  feat_preprocess = pickle.load(open(o.feat_preprocess_file, "rb"))
else:
  logging.error(" feat_preprocess.pkl not provided. Exiting with 1");
  sys.exit(1)

## Load cmvn ##
trn_cmvn = CMVN(feat_preprocess, o.utt2spk_file, o.cmvn_scp)
##

# creating theano neural network
logging.info(" Loading nnet from %s", nnet_in_file)
nnet = NeuralNet()

nnet.load_layers_frm_file(nnet_in_file)
X_ = T.matrix("X")
nnet.link_IO(X_)

logging.info(" Creating theano functions")
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


strm_indices = o.strm_indices
lr = o.learn_rate
(segment_buffer_size, batch_size) = (o.segment_buffer_size, o.batch_size) 

error = accuracy = n = 0.0
for ii in range(o.iters_per_epoch):
  logging.info("iters_per_epoch iter%d", ii);
  with kaldi_io.KaldiScpReader(trn_scp, feature_preprocess.full_preprocess, reader_args=[feat_preprocess, trn_cmvn.utt2spk_dict, trn_cmvn.cmvn_dict]) as trn_data_it:
    for mm, segment_buffer in enumerate(utils.isplit_every(features.fea_lab_pair_iter(trn_data_it, trn_lab_dict), segment_buffer_size)):
      noncum_error = noncum_accuracy = noncum_n = 0.0
      for X, t in apply_mask_to_minibatch(utils.segment_buffer_to_minibatch_iter(segment_buffer, batch_size), strm_indices):
        err, acc = train(X, t, lr)
        error += err; accuracy += acc; n += len(X); 
      noncum_error += err; noncum_accuracy += acc; noncum_n += len(X)
      logging.info("%f | %f | %f | %f | %f", error / n, accuracy /n, noncum_error / noncum_n, noncum_accuracy / noncum_n, lr)
(trn_error, trn_accuracy) = (error / n, accuracy / n)

nnet.save_weights(nnet_out_file)

#create done file
if o.done_file != "":
  done_fd = open(o.done_file, "w")
  done_fd.write("--nSamples=%d\n--train-error=%f\n--train-accuracy=%f\n--learn-rate=%f\n" %(n, error, accuracy, lr))
  done_fd.close()
  
sys.exit(0)

