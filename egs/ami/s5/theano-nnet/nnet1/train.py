#!/usr/bin/env python
import sys, os, logging, numpy as np
#os.environ['THEANO_FLAGS']='nvcc.flags=-arch=sm_30,mode=FAST_RUN,device=gpu,floatX=float32'
os.environ['THEANO_FLAGS']='device=gpu,floatX=float32'
import theano, theano.tensor as T

import scipy.io.wavfile
import itertools
import shutil
import cPickle as pickle

import kaldi_io, utils, features, parse_config
from neuralnet import NeuralNet
import neuralnet
import feature_preprocess
from feature_preprocess import FeaturePreprocess
from feature_preprocess import CMVN
from collections import OrderedDict

import mkl
mkl.set_num_threads(1)
import numexpr
numexpr.set_num_threads(1)

np.random.seed(42)
##################################

from optparse import OptionParser
usage = "%prog [options] <trn-scp> <cv-scp> <trn-labels> <cv-labels> <outdir>"
parser = OptionParser(usage)

parser.add_option('--config', dest="config",
                  help="Configuration file to read (this option may be repeated) [default: %default]",
                  default="", type='string')

parser = parse_config.theano_nnet_parse_opts(parser)

(o, args) = parser.parse_args()
# options specified in config overides command line
if o.config != "": (o, args) = parse_config.parse_config(parser, o.config)

if len(args) != 5:
  parser.print_help()
  sys.exit(1)

(trn_scp, cv_scp, trn_lab, cv_lab, outdir) = (args[0], args[1], args[2], args[3], args[4])
utils.mkdir_p(outdir)
utils.mkdir_p(outdir+"/nnet")

## Create log file
logging.basicConfig(filename=outdir+'/train.log', format='%(asctime)s: %(message)s', level=logging.INFO)
logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))

## Check final_nnet.pkl exist, if yes quit
if os.path.exists(outdir+"/final_nnet.pklz"):
  logging.info("Skipping training as %s/final_nnet.pklz exists", outdir)
  sys.exit(0)

## Load labels ##
logging.info("Loading train labels from %s", trn_lab)
trn_lab_dict = utils.labels_ascii_to_dict(trn_lab)
logging.info("Loading cv labels from %s", cv_lab)
cv_lab_dict  = utils.labels_ascii_to_dict(cv_lab)
##

##
feat_preprocess = FeaturePreprocess(o)

## Load cmvn ##
trn_cmvn = CMVN(feat_preprocess, o.trn_utt2spk_file, o.trn_cmvn_scp)
cv_cmvn = CMVN(feat_preprocess, o.cv_utt2spk_file, o.cv_cmvn_scp)
##

## Check if to start from initial or middle
last_iter, last_cv_error, last_learn_rate, halving, best_nnet_file = utils.theano_nnet_start(outdir, o)
if last_iter != 0:
  logging.info("last_iter=%d, last_cv_error=%f, last_learn_rate=%f, halving=%d, best_nnet_file=%s", last_iter, last_cv_error, last_learn_rate, halving, best_nnet_file)
  # creating theano neural network
  X_ = T.matrix("X")
  nnet = NeuralNet()
  nnet.Load(X_, best_nnet_file)

  #load feat_preprocess.pkl, it should exist
  if os.path.exists(outdir+"/feat_preprocess.pkl") == False:
    logging.info("Expects %s/feat_preprocess.pkl . Something might be wrong, better to restart from begining", outdir);
    sys.exit(1)
  logging.info("Loading feat preprocess from %s/feat_preprocess.pkl", outdir)
  feat_preprocess = pickle.load(open(outdir+"/feat_preprocess.pkl", "rb"))

else:
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
  input_mean = (F/N).astype(T.config.floatX)
  input_std  = np.sqrt(S/N - input_mean**2).astype(T.config.floatX)

  #put it in feat_preprocess
  (feat_preprocess.glob_mean, feat_preprocess.glob_std) = (input_mean, input_std)

  # creating theano neural network
  logging.info("Creating and initializing NN")
  nnet = NeuralNet()
  nnet.initialize_from_proto(o.nnet_proto)
  #nnet.set_name('dnn')
  
  # save feat_preprocess.pkl
  logging.info("Saving feat preprocess to %s/feat_preprocess.pkl", outdir)
  # save feature_preprocess, for further usage 
  pickle.dump(feat_preprocess, open(outdir+"/feat_preprocess.pkl", "wb"))

  # save initial net
  logging.info("Saving initial nnet to %s/nnet_initial.pklz", outdir)
  nnet.save_weights(outdir+"/nnet_initial.pklz")
  nnet_best_file = outdir+"/nnet_initial.pklz"

logging.info("Creating theano functions")
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

xentropy = theano.function(inputs=[X_, T_], outputs=[cost_, acc_])

if last_iter == 0:
  ## Initial CV error
  logging.info("Evaluating on CV")
  error = accuracy = n = 0.0
  with kaldi_io.KaldiScpReader(cv_scp, feature_preprocess.full_preprocess, reader_args=[feat_preprocess, cv_cmvn.utt2spk_dict, cv_cmvn.cmvn_dict]) as cv_data_it:
    for ii, (X, t) in enumerate(features.fea_lab_pair_iter(cv_data_it, cv_lab_dict)):
      err, acc = xentropy(X, t)
      error += err; accuracy += acc; n += len(X)
  logging.info("%d | %f | %f", n, error / n, accuracy / n)
  last_cv_error = error / n

(start_iter, max_iters) = (last_iter+1, o.max_iters)
(lr, halving_factor) = (last_learn_rate, o.halving_factor)
(segment_buffer_size, batch_size) = (o.segment_buffer_size, o.batch_size) 
(start_halving_impr, end_halving_impr) = (o.start_halving_impr, o.end_halving_impr)

# start training
for kk in range(start_iter, max_iters):
  if halving: lr = lr * halving_factor
  np.random.seed(42)
  logging.info("Training epoch: %02d, learning rate: %f", kk, lr)
  error = accuracy = n = 0.0
  with kaldi_io.KaldiScpReader(trn_scp, feature_preprocess.full_preprocess, reader_args=[feat_preprocess, trn_cmvn.utt2spk_dict, trn_cmvn.cmvn_dict]) as trn_data_it:
    for mm, segment_buffer in enumerate(utils.isplit_every(features.fea_lab_pair_iter(trn_data_it, trn_lab_dict), segment_buffer_size)):
      noncum_error = noncum_accuracy = noncum_n = 0.0
      for X, t in utils.segment_buffer_to_minibatch_iter(segment_buffer, batch_size):
        err, acc = train(X, t, lr)
        error += err; accuracy += acc; n += len(X)
        noncum_error += err; noncum_accuracy += acc; noncum_n += len(X)
      logging.info("%f | %f | %f | %f", error / n, accuracy /n, noncum_error / noncum_n, noncum_accuracy / noncum_n)
  (trn_error, trn_accuracy) = (error / n, accuracy / n)

  this_nnet_file_tmp_name = "%s/nnet/nnet_epoch%02d.pklz" %(outdir, kk)
  nnet.save_weights(this_nnet_file_tmp_name)

  logging.info("Evaluating on CV")
  error = accuracy = n = 0.0
  with kaldi_io.KaldiScpReader(cv_scp, feature_preprocess.full_preprocess, reader_args=[feat_preprocess, cv_cmvn.utt2spk_dict, cv_cmvn.cmvn_dict]) as cv_data_it:
    for ii, (X, t) in enumerate(features.fea_lab_pair_iter(cv_data_it, cv_lab_dict)):
      err, acc = xentropy(X, t)
      error += err; accuracy += acc; n += len(X)
  logging.info("%d | %f | %f", n, error / n, accuracy / n)
  (cv_error, cv_accuracy) = (error / n, accuracy /n)

  # change the above nnet 
  this_nnet_file = "%s/nnet/nnet_epoch%02d_learnrate%f_tr%f_cv%f.pklz" %(outdir, kk, lr, trn_error, cv_error)
  logging.info("Saving epoch%d nnet to %s", kk, this_nnet_file)
  shutil.move(this_nnet_file_tmp_name, this_nnet_file)

  done_fd = open("%s/.done_epoch%d" %(outdir, kk), "w")
  done_fd.write("--train-error=%f\n--train-accuracy=%f\n--cv-error=%f\n--cv-accuracy=%f\n--learn-rate=%f\n" %(trn_error, trn_accuracy, cv_error, cv_accuracy, lr))
  done_fd.close()

  # accept or reject new parameters (based on objective function)  
  loss_prev = last_cv_error
  loss_new  = cv_error
  if loss_new < loss_prev:
    last_cv_error = loss_new
    best_nnet_file = this_nnet_file
    best_nnet_fd = open("%s/.best_nnet" %(outdir), "w")
    best_nnet_fd.write("%s" %(best_nnet_file))
    best_nnet_fd.close()
  else:
    # mv 
    shutil.move(this_nnet_file, this_nnet_file+"_rejected")
    logging.info("nnet rejected %s_rejected" %(this_nnet_file))

    # revert params
    nnet.set_weights_frm_file(best_nnet_file)
    logging.info("reverting params to %s" %(best_nnet_file))
    
  # stopping criterion
  rel_impr = (loss_prev - last_cv_error)/loss_prev
  if halving == 1 and rel_impr < end_halving_impr:
    logging.info("finished, too small rel. improvement %f", rel_impr)
    break

  # start annealing when improvement is low
  if rel_impr < start_halving_impr:
    halving = 1
    open(outdir+"/.halving", 'a').close()

# softlink to final_nnet.pkl
old_dir = os.path.abspath(os.getcwd())
os.chdir(outdir)
rel_best_file = "/".join(best_nnet_file.split("/")[-2:])
soft_link_cmd="ln -s %s final_nnet.pklz" %(rel_best_file)
os.system(soft_link_cmd)
os.chdir(old_dir)

if os.path.exists(outdir+"/final_nnet.pklz"):
  logging.info("Succeeded training the Neural Network : %s", outdir+"/final_nnet.pklz")
else:
  logging.info("Error training neural network...")
  sys.exit(1)

sys.exit(0)

