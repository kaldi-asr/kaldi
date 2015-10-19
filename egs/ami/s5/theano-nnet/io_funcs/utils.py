#!/usr/bin/env python

import os, errno, sys
import numpy as np
import scipy.linalg as spl
import numexpr as ne
import logging
import kaldi_io
import itertools
import tarfile
import Queue
from threading import Thread

from optparse import OptionParser

def mkdir_p(path):
    try:
        os.makedirs(path)
    except OSError as exc:
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else: raise

def segment_buffer_to_minibatch_iter(segment_buffer, minibatch_size=512):
  # make sure that all data matrices in each tupple are of the same length
  seg_lengths=zip(*[[len(e) for e in t] for t in segment_buffer])
  assert seg_lengths[1:]==seg_lengths[:-1]
  
  buffers = [np.concatenate(buf) for buf in zip(*segment_buffer)]
  shuffle = np.random.permutation(len(buffers[0]))
  n_minibatches = len(shuffle) / minibatch_size
  for minibatch in zip(*[np.array_split(buf.take(shuffle, axis=0), n_minibatches) for buf in buffers]):
    yield minibatch

import itertools

def isplit_every(it, n, drop_uncomplete_last=True):
  it = iter(it)
  while True:
    buffer = list(itertools.islice(it, n))
    if not buffer or len(buffer) < n and drop_uncomplete_last:
      break
    yield buffer

def entropy(postgram, axis=1):
  # Added by Harish
  return np.sum(-1*postgram*np.log2(postgram), axis=axis)

def theano_nnet_start(nnet_dir, o):

  # starting iteration
  last_iter = 0
  last_cv_error = np.inf
  last_learn_rate = o.learn_rate
  halving = 0
  best_nnet_file = ""
  for kk in range(o.max_iters, 0, -1):
    done_file = "%s/.done_epoch%d" %(nnet_dir, kk)
    if os.path.exists(done_file):
      last_iter = kk

      po = OptionParser()
      po = nnet_donefile_options(po)
      (done_o, a) = po.parse_args(parse_config_dict(done_file))

      last_learn_rate = done_o.learn_rate
      last_cv_error = done_o.cv_error
      halving_file = "%s/.halving" %(nnet_dir)
      if os.path.exists(halving_file):
        halving = 1
      
      if os.path.exists(nnet_dir+"/.best_nnet"):
        best_nnet_file = open(nnet_dir+"/.best_nnet", "r").readline().rstrip()

      break

  return (last_iter, last_cv_error, last_learn_rate, halving, best_nnet_file)


def theano_nnet_start_ExpLrSched(nnet_dir, o):

  # starting iteration
  last_iter = 0
  last_learn_rate = o.learn_rate
  nSamples = 0.0
  best_nnet_file = ""
  for kk in range(o.max_iters, 0, -1):
    done_file = "%s/.done_epoch%d" %(nnet_dir, kk)
    if os.path.exists(done_file):
      last_iter = kk

      po = OptionParser()
      po = nnet_donefile_options(po)
      (done_o, a) = po.parse_args(parse_config_dict(done_file))

      last_learn_rate = done_o.learn_rate
      nSamples = done_o.nSamples

      if os.path.exists(nnet_dir+"/.best_nnet"):
        best_nnet_file = open(nnet_dir+"/.best_nnet", "r").readline().rstrip()

  return (last_iter, last_learn_rate, nSamples, best_nnet_file)

def labels_ascii_to_dict(labels_ascii_file):

  labels_dict = {}
  for ii, line in enumerate(open(labels_ascii_file, "r")):
    line = line.rstrip()
    labels_dict[line.split()[0]] = np.asarray(map(lambda x: int(x), line.split()[1:])).astype(np.int16)

  return labels_dict

import itertools

if(__name__=="__main__"):
    pass
