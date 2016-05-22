#!/usr/bin/env python
import sys, os, logging, numpy as np
#os.environ['THEANO_FLAGS']='nvcc.flags=-arch=sm_30,mode=FAST_RUN,device=gpu,floatX=float32'
os.environ['THEANO_FLAGS']='device=cpu,floatX=float32'
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
usage = "%prog nnet_proto nnet_initialize"
parser = OptionParser(usage)
parser = parse_config.theano_nnet_parse_opts(parser)

(o, args) = parser.parse_args()

if len(args) != 2:
  parser.print_help()
  sys.exit(1)

(nnet_proto, nnet_initial) = (args[0], args[1])

nnet = NeuralNet()
nnet.initialize_from_proto(nnet_proto)
nnet.save_weights(nnet_initial)

sys.exit(0)

