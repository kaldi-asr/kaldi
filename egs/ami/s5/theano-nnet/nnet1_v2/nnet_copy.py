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
usage = "%prog [options] inp_model out_model"
parser = OptionParser(usage)

parser.add_option('--config', dest="config",
                  help="Configuration file to read (this option may be repeated) [default: %default]",
                  default="", type='string')

parser.add_option('--remove-first-components', dest="remove_first_components",
                  help="Remove N first layers Components from the Nnet (int, default = 0) [default: %default]",
                  default="0", type='string')

parser.add_option('--remove-last-components', dest="remove_last_components",
                  help="Remove N last layers Components from the Nnet (int, default = 0) [default: %default]",
                  default="0", type='string')

parser = parse_config.theano_nnet_parse_opts(parser)

(o, args) = parser.parse_args()
# options specified in config overides command line
if o.config != "": (o, args) = parse_config.parse_config(parser, o.config)

if len(args) != 2:
  parser.print_help()
  sys.exit(1)

(inp_nnet_file, out_nnet_file) = (args[0], args[1])
o.remove_first_components = int(o.remove_first_components)
o.remove_last_components = int(o.remove_last_components)

## Create log file
logging.basicConfig(stream=sys.stdout, format='%(asctime)s: %(message)s', level=logging.INFO)

logging.info(" Running as %s ", sys.argv[0])
logging.info(" %s", " ".join(sys.argv))

# creating theano neural network
logging.info(" Loading nnet from %s", inp_nnet_file)
out_nnet = NeuralNet()
out_nnet.load_layers_frm_file(inp_nnet_file)

for i in xrange(o.remove_first_components):
  out_nnet.remove_first_layer()

for i in xrange(o.remove_last_components):
  out_nnet.remove_last_layer()

out_nnet.save_weights(out_nnet_file)

sys.exit(0)



