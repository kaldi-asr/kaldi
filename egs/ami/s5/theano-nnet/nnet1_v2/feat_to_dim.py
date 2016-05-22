#!/usr/bin/env python
import sys, os, logging, numpy as np

# import kaldi_io, utils, features, parse_config
# from neuralnet import NeuralNet
# import neuralnet
# import feature_preprocess
# from feature_preprocess import FeaturePreprocess
# from feature_preprocess import CMVN

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import parse_config
from io_funcs import kaldi_io, utils
from feature_funcs import features, feature_preprocess

from neuralnet import NeuralNet
import neuralnet
from feature_funcs.feature_preprocess import FeaturePreprocess, CMVN

from optparse import OptionParser
usage = "%prog [options] <feats-scp>"
parser = OptionParser(usage)

parser.add_option('--config', dest="config",
                  help="Configuration file to read (this option may be repeated) [default: %default]",
                  default="", type='string')

parser = parse_config.theano_nnet_parse_opts(parser)

(o, args) = parser.parse_args()
# options specified in config overides command line
if o.config != "": (o, args) = parse_config.parse_config(parser, o.config)

if len(args) != 1:
  parser.print_help()
  sys.exit(1)

feats_scp = args[0]

##
feat_preprocess = FeaturePreprocess(o)

## Load cmvn ##
cmvn = CMVN(feat_preprocess, o.utt2spk_file, o.cmvn_scp)


with kaldi_io.KaldiScpReader(feats_scp, feature_preprocess.full_preprocess, reader_args=[feat_preprocess, cmvn.utt2spk_dict, cmvn.cmvn_dict]) as data_it:
  for ii, (X, utt) in enumerate(data_it):
    print X.shape[1]
    data_it.close()
    sys.exit(0)

