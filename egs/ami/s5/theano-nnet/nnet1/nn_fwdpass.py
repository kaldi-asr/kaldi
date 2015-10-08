#!/usr/bin/env python
import sys, os, logging, numpy as np
os.environ['THEANO_FLAGS']='nvcc.flags=-arch=sm_30,mode=FAST_RUN,device=cpu,floatX=float32' #since fwdpass
import theano, theano.tensor as T

# import pytel.htk, pytel.utils, pytel.features, pytel.kaldi_io
# import mypytel.kaldi_io, mypytel.utils

import scipy.io.wavfile
import tarfile
import itertools
import StringIO
import shutil
import cPickle as pickle

import kaldi_io, utils, features
from neuralnet import NeuralNet
import feature_preprocess
from feature_preprocess import FeaturePreprocess
from feature_preprocess import CMVN
from PdfPrior import PdfPrior

import mkl
mkl.set_num_threads(1)
import numexpr
numexpr.set_num_threads(1)

np.random.seed(42)
##################################

from optparse import OptionParser
usage = "%prog [options] <nnet-dir> <data-dir>"
parser = OptionParser(usage)

parser.add_option('--config', dest="config",
                  help="Configuration file to read (this option may be repeated) [default: %default]",
                  default="", type='string')

parser.add_option('--prior-floor', dest='prior_floor',
                  help="Flooring constatnt for prior probability (i.e. label rel. frequency) [default:%default]",
                  default=1e-10, type=float)
parser.add_option('--prior-scale', dest='prior_scale',
                  help="Scaling factor to be applied on pdf-log-priors [default:%default]",
                  default=1.0, type=float)

parser.add_option('--no-softmax', dest='no_softmax', 
                  help='No softmax on MLP output (or remove it if found), the pre-softmax activations will be used as log-likelihoods, log-priors will be subtracted [default: %default]', 
                  default="false", type=str)
parser.add_option('--apply-log', dest='apply_log', 
                  help='Transform MLP output to logscale [default: %default]', 
                  default="false", type=str)

parser.add_option('--feat-preprocess', dest='feat_preprocess', 
                  help='Feature transform in front of main network [default: %default]', 
                  default="", type=str)
parser.add_option('--utt2spk-file', dest="utt2spk_file",
                  help="kaldi utt2spk [default: %default]",
                  default="", type=str)
parser.add_option('--cmvn-scp', dest="cmvn_scp",
                  help="kaldi utt2spk [default: %default]",
                  default="", type=str)

parser.add_option('--class-frame-counts', dest='class_frame_counts',
                  help='Vector with frame-counts of pdfs to compute log-priors. (priors are typically subtracted from log-posteriors or pre-softmax activations) [default: %default]', 
                  default="", type=str)

(o, args) = parser.parse_args()
# options specified in config overides command line
if o.config != "": (o, args) = utils.parse_config(parser, o.config)

if len(args) != 2:
  parser.print_help()
  sys.exit(1)

(nnet_dir, data_dir) = (args[0], args[1])

## Create log file
logging.basicConfig(format='%(asctime)s: %(message)s', level=logging.INFO)
logging.getLogger().addHandler(logging.StreamHandler(sys.stderr))

## Check final_nnet.pkl exist, if yes quit
if not os.path.exists(nnet_dir+"/final_nnet.pklz"):
  logging.error("%s does not exist. Skipping fwdpass", nnet_dir+"/final_nnet.pklz")
  sys.exit(1)

## feature preprocessing ##
if o.feat_preprocess != "":
  logging.info("Loading feature preprocess from %s", o.feat_preprocess)
  feat_preprocess = pickle.load(open(o.feat_preprocess, "rb"))
else:
  if os.path.exists(nnet_dir+"/feat_preprocess.pkl"):
    logging.info("Loading feature preprocess from %s/feat_preprocess.pkl", nnet_dir)
    feat_preprocess = pickle.load(open(nnet_dir+"/feat_preprocess.pkl", "rb"))
  else:
    logging.error("feat_preprocess.pkl not provided. Exiting with 1");
    sys.exit(1)

# prior opts
pdf_prior = PdfPrior(o)

## Load cmvn ##
data_cmvn = CMVN(feat_preprocess, o.utt2spk_file, o.cmvn_scp)
##

## Check if to start from initial or middle
nnet_file = nnet_dir+"/final_nnet.pklz"
nnet = NeuralNet()

nnet.load_layers_frm_file(nnet_file)
X_ = T.matrix("X")
nnet.link_IO(X_)

logging.info("Creating theano functions")
(X_, Y_, params_) = (nnet.X, nnet.Y, nnet.params)

if o.no_softmax == "true" and o.apply_log == "true":
  logging.error("Cannot use both --apply-log=true --no-softmax=true, use only one of the two!")
  sys.exit(1)

if o.apply_log == "true":
  Y_ = T.log(Y_ + 1e-20) #avoid log(0)

if o.no_softmax == "true":
  logging.info("Removing Softmax from %s", nnet_dir+"/final_nnet.pklz")
  Y_ = Y_.owner.inputs[0]


fwdpass_fn = theano.function(inputs=[X_], outputs=Y_)

data_scp = data_dir+"/feats.scp"
if not os.path.exists(data_scp):
  logging.error("%s doesn't exist", data_scp)
  sys.exit(1)

with kaldi_io.KaldiScpReader(data_scp, feature_preprocess.full_preprocess, reader_args=[feat_preprocess, data_cmvn.utt2spk_dict, data_cmvn.cmvn_dict]) as data_it:
  for ii, (X, utt) in enumerate(data_it):
    logging.info("processing utt = %s", utt)
    Y = fwdpass_fn(X)

    if o.class_frame_counts != "":
      if np.min(Y) >= 0.0 and np.max(Y) <= 1.0:
        logging.error("Subtracting log-prior on 'probability-like' data in range [0..1]\
(Did you forget --no-softmax=true or --apply-log=true ?)")
        sys.exit(1)
      Y = pdf_prior.SubtractOnLogpost(Y)

    kaldi_io.write_stdout_ascii(Y, utt)

