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
##################################

from optparse import OptionParser
usage = "%prog [options] <cv-scp> <cv-labels> <nnet>"
parser = OptionParser(usage)

parser.add_option('--config', dest="config",
                  help="Configuration file to read (this option may be repeated) [default: %default]",
                  default="", type='string')

parser.add_option('--feat-preprocess', dest="feat_preprocess_file",
                  help="Feature preprocess file [default: %default]",
                  default="", type='string')

parser.add_option('--done-file', dest="done_file",
                  help="Done file to dump [default: %default]",
                  default="", type='string')



parser = parse_config.theano_nnet_parse_opts(parser)

(o, args) = parser.parse_args()
# options specified in config overides command line
if o.config != "": (o, args) = parse_config.parse_config(parser, o.config)

if len(args) != 3:
  parser.print_help()
  sys.exit(1)

(cv_scp, cv_lab_file, nnet_file) = (args[0], args[1], args[2])

## Create log file
logging.basicConfig(stream=sys.stdout, format='%(asctime)s: %(message)s', level=logging.INFO)

logging.info(" Running as %s ", sys.argv[0])
logging.info(" %s", " ".join(sys.argv))

## Load labels ##
logging.info(" Loading labels from %s", cv_lab_file)
cv_lab_dict = utils.labels_ascii_to_dict(cv_lab_file)
##

## feature preprocessing ##
if o.feat_preprocess_file != "":
  logging.info(" Loading feature preprocess from %s", o.feat_preprocess_file)
  feat_preprocess = pickle.load(open(o.feat_preprocess_file, "rb"))
else:
  logging.error(" feat_preprocess.pkl not provided. Exiting with 1");
  sys.exit(1)

## Load cmvn ##
cv_cmvn = CMVN(feat_preprocess, o.cv_utt2spk_file, o.cv_cmvn_scp)
##

# creating theano neural network
logging.info(" Loading nnet from %s", nnet_file)
nnet = NeuralNet()

nnet.load_layers_frm_file(nnet_file)
X_ = T.matrix("X")
nnet.link_IO(X_)

logging.info(" Creating theano functions")
(X_, Y_) = (nnet.X, nnet.Y)
lr_ = T.scalar()
T_ = T.ivector("T")
cost_ = T.nnet.categorical_crossentropy(Y_, T_).sum()
acc_ = T.eq(T.argmax(Y_, axis=1), T_).sum()

xentropy = theano.function(inputs=[X_, T_], outputs=[cost_, acc_])

error = accuracy = n = 0.0

with kaldi_io.KaldiScpReader(cv_scp, feature_preprocess.full_preprocess, reader_args=[feat_preprocess, cv_cmvn.utt2spk_dict, cv_cmvn.cmvn_dict]) as cv_data_it:
  for ii, (X, t) in enumerate(features.fea_lab_pair_iter(cv_data_it, cv_lab_dict)):
    err, acc = xentropy(X, t)
    error += err; accuracy += acc; n += len(X)
logging.info("%d | %f | %f", n, error / n, accuracy / n)

(cv_error, cv_accuracy) = (error / n, accuracy / n)
#create done file
if o.done_file != "":
  done_fd = open(o.done_file, "w")
  done_fd.write("--nSamples=%d\n--cv-error=%f\n--cv-accuracy=%f\n" %(n, error, accuracy))
  done_fd.close()

  import time
  time.sleep(60)
  
sys.exit(0)


