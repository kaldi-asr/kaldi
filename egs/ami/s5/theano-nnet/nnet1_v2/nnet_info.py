#!/usr/bin/env python
import sys, os, logging, numpy as np
##os.environ['THEANO_FLAGS']='nvcc.flags=-arch=sm_30,mode=FAST_RUN,device=gpu,floatX=float32'
#os.environ['THEANO_FLAGS']='device=cpu,floatX=float32'
#import theano, theano.tensor as T

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
usage = "%prog [options] inp_nnet"
parser = OptionParser(usage)
(o, args) = parser.parse_args()

if len(args) != 1:
  parser.print_help()
  sys.exit(1)

inp_nnet_file = args[0]

inp_nnet = NeuralNet()
inp_nnet.load_layers_frm_file(inp_nnet_file)

#print the info
inp_nnet.Info()

sys.exit(0)



