#!/usr/bin/env python
import sys, os, logging, numpy as np

import scipy.io.wavfile
import itertools, shutil
import cPickle as pickle
from collections import OrderedDict

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import parse_config
from io_funcs import kaldi_io, utils
from feature_funcs import features, feature_preprocess

from feature_funcs.feature_preprocess import FeaturePreprocess, CMVN


import mkl
mkl.set_num_threads(1)
import numexpr
numexpr.set_num_threads(1)

np.random.seed(42)
##################################

from optparse import OptionParser
usage = "%prog [options] <trn-scp> <feat-preprocess-file>"
parser = OptionParser(usage)

parser.add_option('--config', dest="config",
                  help="Configuration file to read (this option may be repeated) [default: %default]",
                  default="", type='string')

parser = parse_config.theano_nnet_parse_opts(parser)

(o, args) = parser.parse_args()
# options specified in config overides command line
if o.config != "": (o, args) = parse_config.parse_config(parser, o.config)

if len(args) != 2:
  parser.print_help()
  sys.exit(1)

(trn_scp, feat_preprocess_file) = (args[0], args[1])

## Create log file
logging.basicConfig(stream=sys.stdout, format='%(asctime)s: %(message)s', level=logging.INFO)

logging.info(" Running as %s ", sys.argv[0])
logging.info(" %s", " ".join(sys.argv))

##
feat_preprocess = FeaturePreprocess(o)
##

## Load cmvn ##
trn_cmvn = CMVN(feat_preprocess, o.trn_utt2spk_file, o.trn_cmvn_scp)
##

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
input_mean = (F/N).astype('float32')
input_std  = np.sqrt(S/N - input_mean**2).astype('float32')

#put it in feat_preprocess
(feat_preprocess.glob_mean, feat_preprocess.glob_std) = (input_mean, input_std)

# save feat_preprocess.pkl
logging.info(" Saving feat preprocess to %s", feat_preprocess_file)
# save feature_preprocess, for further usage 
pickle.dump(feat_preprocess, open(feat_preprocess_file, "wb"))

sys.exit(0)

