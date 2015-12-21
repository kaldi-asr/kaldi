#!/usr/bin/env python
import sys, os, logging, numpy as np
os.environ['THEANO_FLAGS']='nvcc.flags=-arch=sm_30,mode=FAST_RUN,device=cpu,floatX=float32' #since fwdpass
import theano, theano.tensor as T


import scipy.io.wavfile
import tarfile, itertools, StringIO, shutil
import cPickle as pickle

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import parse_config
from io_funcs import kaldi_io, utils
from feature_funcs import features, feature_preprocess

from nnet1_v2.neuralnet import NeuralNet
import nnet1_v2.neuralnet
from feature_funcs.feature_preprocess import FeaturePreprocess, CMVN

import compute_mtd

import mkl
mkl.set_num_threads(1)
import numexpr
numexpr.set_num_threads(1)

np.random.seed(42)
##################################

def AdditiveSmoothing(Y, floor=1e-10):

  Y = Y + floor
  S = np.sum(Y, axis=1)
  Y = Y / S[:, np.newaxis]
  
  return Y


##################################

def apply_strm_mask(data_it, strm_indices, comb_num):

  # bin_func = lambda x : ''.join(reversed( [str((x >> i) & 1) for i in range(nstrms)] ) )
  # np.asarray(map(lambda x: int(x), w_strm))

  nstrms=len(strm_indices)  

  bin_str='{:010b}'.format(comb_num)
  bin_str=bin_str[-nstrms:]
  wts = np.asarray(map(lambda x: int(x), bin_str))

  for X, utt in data_it:
    assert strm_indices[-1][-1] == X.shape[1]
    
    #apply weights
    for ii, (st, en) in enumerate(strm_indices):
      X[:, st:en] = X[:, st:en] * wts[ii]

    yield X, utt



def gen_all_comb(X, strm_indices):
  nstrms = len(strm_indices)

  for i in range(1, np.power(2, nstrms)):
    s = map(lambda x: int(x), list(("{:0%db}"%nstrms).format(i)))
    Xc = np.zeros_like(X)
    for ii, (st, en) in enumerate(strm_indices):
      Xc[:,st:en] = X[:,st:en] * s[ii]
    yield Xc, s

##################################
def read_pdf_to_phone_map(pdf_to_pseudo_phone_file):
  vec = np.loadtxt(pdf_to_pseudo_phone_file, dtype=int)
  pdf_to_phone_map = {}
  max_phone_id = 0
  for i in xrange(len(vec)):
    if len(vec[i]) != 2 or vec[i][0] < 0 or vec[i][1] < 0:
      logging.error("Error reading pdf to phone map from %s (bad line %d )", pdf_to_phone_map_file, i)
      sys.exit(1)
    if vec[i][0] >= len(vec):
      logging.error("Pdf-id seems too large: given %d, while expecting a number less than size %d", vec[i][0], len(vec))
      sys.exit(1)
    if vec[i][0] in pdf_to_phone_map.keys():
      logging.error("Pdf-id has been mapped to %d, please keep pdf to phone map unique.", pdf_to_phone_map[vec[i][0]])
      sys.exit(1)

    pdf_to_phone_map[vec[i][0]] = vec[i][1]
    if vec[i][1] > max_phone_id:
      max_phone_id = vec[i][1]
    
  return pdf_to_phone_map, max_phone_id

##################################
def logit(a):
  return np.log(a / (1 - a))

##################################
def compute_phone_post(Y, pdf_to_phone_map, max_phone_id): #Y=state_post

  assert Y.shape[1] == len(pdf_to_phone_map)
  Y_out = np.zeros((Y.shape[0], max_phone_id+1), dtype='float32')
  
  for j in xrange(Y.shape[1]):
    Y_out[:, pdf_to_phone_map[j]] += Y[:, j]

  return Y_out

##################################



from optparse import OptionParser
usage = "%prog [options] <nnet-dir> <data-dir> <out_acc.pklz>"
parser = OptionParser(usage)

parser.add_option('--config', dest="config",
                  help="Configuration file to read (this option may be repeated) [default: %default]",
                  default="", type='string')

parser.add_option('--feat-preprocess', dest='feat_preprocess', 
                  help='Feature transform in front of main network [default: %default]', 
                  default="", type=str)
parser.add_option('--utt2spk-file', dest="utt2spk_file",
                  help="kaldi utt2spk [default: %default]",
                  default="", type=str)
parser.add_option('--cmvn-scp', dest="cmvn_scp",
                  help="kaldi utt2spk [default: %default]",
                  default="", type=str)

parser.add_option('--strm-indices', dest="strm_indices",
                  help="Indices for application of strm wts [default: %default]",
                  default="", type="string")
parser.add_option('--comb-num', dest="comb_num",
                  help="Combination number [default: %default]",
                  default="0", type='string')

parser.add_option('--apply-log', dest="apply_log",
                  help="Ouput log phoneme posts [default: %default]",
                  default="false", type='string')
parser.add_option('--apply-logit', dest="apply_logit",
                  help="Ouput logit of phoneme posts [default: %default]",
                  default="false", type='string')

parser.add_option('--pdf-to-pseudo-phone', dest='pdf_to_pseudo_phone',
                  help="A two column file that maps pdf-id to the corresponding pseudo phone-id. If supplied, outputs the log probabilities for given phones instead of pdf's [default: %default]",
                  default = "", type=str)


(o, args) = parser.parse_args()
# options specified in config overides command line
if o.config != "": (o, args) = utils.parse_config(parser, o.config)

if len(args) != 3:
  parser.print_help()
  sys.exit(1)

(nnet_dir, data_dir, out_acc_pklz) = (args[0], args[1], args[2])

## Create log file
logging.basicConfig(stream=sys.stderr, format='%(asctime)s: %(message)s', level=logging.INFO)

logging.info(" Running as %s ", sys.argv[0])
logging.info(" %s", " ".join(sys.argv))

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

if len(o.strm_indices) != 0:
  logging.info("Using strm indices %s", o.strm_indices)
  a = map(lambda x: int(x), o.strm_indices.split(","))
  o.strm_indices = []
  for st, en in zip(a[0:-1], a[1:]):
    o.strm_indices.append((st, en))

pdf_to_phone_map, max_phone_id = read_pdf_to_phone_map(o.pdf_to_pseudo_phone)

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

fwdpass_fn_for_mstrm = theano.function(inputs=[X_], outputs=Y_)

Y_out_ = Y_
fwdpass_fn = theano.function(inputs=[X_], outputs=Y_out_)

data_scp = data_dir+"/feats.scp"
if not os.path.exists(data_scp):
  logging.error("%s doesn't exist", data_scp)
  sys.exit(1)

strm_indices=o.strm_indices
comb_num=int(o.comb_num)

with kaldi_io.KaldiScpReader(data_scp, feature_preprocess.full_preprocess, reader_args=[feat_preprocess, data_cmvn.utt2spk_dict, data_cmvn.cmvn_dict]) as data_it:
  for ii, (X, utt) in enumerate(apply_strm_mask(data_it, strm_indices, comb_num)):
    logging.info("processing utt = %s, ii= %d", utt, ii)
    Y = compute_phone_post(fwdpass_fn_for_mstrm(X), pdf_to_phone_map, max_phone_id)

    if o.apply_log == "true":
      Y = np.log(Y)
    elif o.apply_logit == "true":
      Y = AdditiveSmoothing(Y, 1e-5)
      Y = logit(Y)

    if ii == 0:
      N = Y.shape[0]
      F = np.sum(Y, axis=0)
      S = np.dot(np.transpose(Y), Y)
    else:
      N = N + Y.shape[0]
      F = F + np.sum(Y, axis=0)
      S = S + np.dot(np.transpose(Y), Y)

#store N, F, S
stats_dict = {}
(stats_dict['N'], stats_dict['F'], stats_dict['S']) = (N, F, S)

import bz2
f = bz2.BZ2File(out_acc_pklz, "wb")
pickle.dump(stats_dict, f)
f.close()

logging.info("Succeded accumalating N, F, S in %s", out_acc_pklz)
sys.exit(0)

# Do EigenAnalysis

# sum_mat = F / float(N)
# sumsq_mat = S / float(N)
# sumsq_mat = sumsq_mat - np.outer(sum_mat, sum_mat)
# (w, v) = np.linalg.eig(sumsq_mat)



