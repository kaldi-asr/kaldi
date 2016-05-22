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

from neuralnet import NeuralNet
import neuralnet
from feature_funcs.feature_preprocess import FeaturePreprocess, CMVN

from PdfPrior import PdfPrior

import compute_mtd

import mkl
mkl.set_num_threads(1)
import numexpr
numexpr.set_num_threads(1)

np.random.seed(42)
##################################
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
def compute_phone_post(Y, pdf_to_phone_map, max_phone_id): #Y=state_post

  assert Y.shape[1] == len(pdf_to_phone_map)
  Y_out = np.zeros((Y.shape[0], max_phone_id+1), dtype='float32')
  
  for j in xrange(Y.shape[1]):
    Y_out[:, pdf_to_phone_map[j]] += Y[:, j]

  return Y_out

##################################
def compute_mdelta(A,y):
  # solve y = Ax using least square method
  (x,residuals,rank,s) = np.linalg.lstsq(A,y)
  # # Mdelta = Mac - Mwc
  # # x[0]: Mwc (within-class MTD)
  # # x[1]: Mac (across-class MTD)
  # print(y.T)
  mdelta = x[1]-x[0]
  # print(x[1],x[0],mdelta)
     # print (mdelta)
  return mdelta

def wrap_compute_mdelta(Y, pri): #phone post or state post
  delta_t = range(1,6)+range(10,81,5)
  m_curve = compute_mtd.compute_mtd(Y, delta_t)

  #fix pri 
  if pri.shape[0] != m_curve.shape[0]:
    new_pri = pri[0:m_curve.shape[0],:]
    pri = new_pri

  this_H = compute_mdelta(pri, m_curve)

  return this_H

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

parser.add_option('--strm-indices', dest="strm_indices",
                  help="Indices for application of strm wts [default: %default]",
                  default="", type="string")

parser.add_option('--pdf-to-pseudo-phone', dest='pdf_to_pseudo_phone',
                  help="A two column file that maps pdf-id to the corresponding pseudo phone-id. If supplied, outputs the log probabilities for given phones instead of pdf's [default: %default]",
                  default = "", type=str)

parser.add_option('--mdelta-prior-file', dest='mdelta_prior_file',
                  help="Prior file for mdelta computaion [default: %default]",
                  default = "", type=str)


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

if len(o.strm_indices) != 0:
  logging.info("Using strm indices %s", o.strm_indices)
  a = map(lambda x: int(x), o.strm_indices.split(","))
  o.strm_indices = []
  for st, en in zip(a[0:-1], a[1:]):
    o.strm_indices.append((st, en))

pdf_to_phone_map, max_phone_id = read_pdf_to_phone_map(o.pdf_to_pseudo_phone)
pri = np.loadtxt(o.mdelta_prior_file)

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

fwdpass_fn_for_mstrm = theano.function(inputs=[X_], outputs=Y_)

Y_out_ = Y_
if o.apply_log == "true":
  Y_out_ = T.log(Y_out_ + 1e-20) #avoid log(0)

if o.no_softmax == "true":
  logging.info("Removing Softmax from %s", nnet_dir+"/final_nnet_cpu.pkl")
  Y_out_ = Y_out_.owner.inputs[0]

fwdpass_fn = theano.function(inputs=[X_], outputs=Y_out_)

data_scp = data_dir+"/feats.scp"
if not os.path.exists(data_scp):
  logging.error("%s doesn't exist", data_scp)
  sys.exit(1)

num_strms = len(o.strm_indices)

with kaldi_io.KaldiScpReader(data_scp, feature_preprocess.full_preprocess, reader_args=[feat_preprocess, data_cmvn.utt2spk_dict, data_cmvn.cmvn_dict]) as data_it:

  for ii, (X, utt) in enumerate(data_it):

    logging.info("processing utt = %s", utt)
    goodness_scores = np.zeros(2**num_strms - 1)
    comb_ids = []
    Xc_list = []

    for c, (Xc, comb_id) in enumerate(gen_all_comb(X, o.strm_indices)):
      #goodness_scores[c] = np.mean(compute_mtd.compute_mtd(fwdpass_fn_for_mstrm(Xc)))
      goodness_scores[c] = np.mean(wrap_compute_mdelta(compute_phone_post(fwdpass_fn_for_mstrm(Xc), pdf_to_phone_map, max_phone_id), pri))
      comb_ids.append(comb_id)
      Xc_list.append(Xc)

    logging.info("print best comb number=%d, with weights %s", np.argmax(goodness_scores)+1, ' '.join(str(x) for x in comb_ids[np.argmax(goodness_scores)]))
    Xc_best = Xc_list[np.argmax(goodness_scores)]
    
    Y = fwdpass_fn(Xc_best)
    if o.class_frame_counts != "":
      if np.min(Y) >= 0.0 and np.max(Y) <= 1.0:
        logging.error("Subtracting log-prior on 'probability-like' data in range [0..1]\
(Did you forget --no-softmax=true or --apply-log=true ?)")
        sys.exit(1)
      Y = pdf_prior.SubtractOnLogpost(Y)

    kaldi_io.write_stdout_ascii(Y, utt)

