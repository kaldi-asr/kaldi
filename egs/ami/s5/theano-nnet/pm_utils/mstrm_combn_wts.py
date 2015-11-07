#!/usr/bin/env python
import sys, os, logging, numpy as np
import theano, theano.tensor as T
import numpy.matlib

import scipy.io.wavfile
import itertools, shutil
import cPickle as pickle
from collections import OrderedDict

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from io_funcs import kaldi_io, utils

import mkl
mkl.set_num_threads(1)
import numexpr
numexpr.set_num_threads(1)

np.random.seed(42)
##################################

def get_strm_mask(data_it, nstrms, comb_num):

  # bin_func = lambda x : ''.join(reversed( [str((x >> i) & 1) for i in range(nstrms)] ) )
  # np.asarray(map(lambda x: int(x), w_strm))

  bin_str='{:010b}'.format(comb_num)
  bin_str=bin_str[-nstrms:]
  wts = np.asarray(map(lambda x: int(x), bin_str))

  #repmat wts
  for X, utt in data_it:
    Wts = np.matlib.repmat(wts, X.shape[0], 1)

    yield Wts, utt

from optparse import OptionParser
usage = "%prog [options] <inp-scp>"
parser = OptionParser(usage)

parser.add_option('--nstrms', dest="nstrms",
                  help="Number of streams [default: %default]",
                  default="0", type='string')

parser.add_option('--comb-num', dest="comb_num",
                  help="combination number [1..31] [default: %default]",
                  default="0", type='string')

(o, args) = parser.parse_args()

if len(args) != 1:
  parser.print_help()
  sys.exit(1)

inp_scp = args[0]

## Create log file
logging.basicConfig(stream=sys.stderr, format='%(asctime)s: %(message)s', level=logging.INFO)

logging.info(" Running as %s ", sys.argv[0])
logging.info(" %s", " ".join(sys.argv))

nstrms = int(o.nstrms)
comb_num = int(o.comb_num)

if nstrms == 0 or comb_num == 0:
  logging.error("ERROR : specify correct stream number and correct comb_num")
  sys.exit(1)

with kaldi_io.KaldiScpReader(inp_scp) as data_it:
  for ii, (Wts, utt) in enumerate(get_strm_mask(data_it, nstrms, comb_num)):
    logging.info("processing utt = %s", utt)
    kaldi_io.write_stdout_ascii(Wts, utt)

sys.exit(0)


