#!/usr/bin/python

import sys, os, logging, numpy as np
import cPickle as pickle

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from io_funcs import kaldi_io, utils

import numexpr
numexpr.set_num_threads(1)

from optparse import OptionParser
usage = "%prog [options] <mse_dir/feats.scp> <scores.pkl>"
parser = OptionParser(usage)

parser.add_option('--uttLevel', dest='uttLevel',
                  help='if true, same weights for all frames in an utterance [default: %default]',
                  default="true", action='store_true');

(o, args) = parser.parse_args()
if len(args) != 2:
  parser.print_help()
  sys.exit(1)

(mse_feats_scp, scores_pkl) = (args[0], args[1])

scores={}
with kaldi_io.KaldiScpReader(mse_feats_scp) as data_it:
  for ii, (X, utt) in enumerate(data_it):
    #0th dim is per-frame mse
    H = X[:,0]

    inv_H = 1./np.asarray(H)
    inv_H = np.power(inv_H, 2) #per-frame scores
  
    if o.uttLevel == "true":
      inv_H = np.mean(inv_H)

    scores[utt] = inv_H
    

#pickle dump
import bz2
f = bz2.BZ2File(scores_pkl, "wb")
pickle.dump(scores, f)
f.close()



