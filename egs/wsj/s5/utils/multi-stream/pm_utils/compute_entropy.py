#!/usr/bin/env python
import sys, os, logging, numpy as np
import cPickle as pickle
import compute_mtd

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from io_funcs import kaldi_io, utils

try:
  import mkl
  mkl.set_num_threads(1)
except ImportError:
  pass

import numexpr
numexpr.set_num_threads(1)

import scipy.stats

##################################
def compute_entropy(X):

  H = []
  for ii, this_post in enumerate(X):
    H.append(scipy.stats.entropy(this_post))

  H = np.asarray(H)

  return H
    
##################################


from optparse import OptionParser
usage = "%prog [options] <post-scp> <out.pkl>"
parser = OptionParser(usage)

parser.add_option('--config', dest="config",
                  help="Configuration file to read (this option may be repeated) [default: %default]",
                  default="", type='string')

(o, args) = parser.parse_args()

if len(args) != 2:
  parser.print_help()
  sys.exit(1)

(post_scp, out_pkl) = (args[0], args[1])

## Create log file
logging.basicConfig(stream=sys.stderr, format='%(asctime)s: %(message)s', level=logging.INFO)

logging.info(" Running as %s ", sys.argv[0])
logging.info(" %s", " ".join(sys.argv))

scores={}
with kaldi_io.KaldiScpReader(post_scp) as data_it:
  for ii, (X, utt) in enumerate(data_it): #X is posteriorgram
    logging.info("processing utt = %s, ii= %d", utt, ii)
    scores[utt] = compute_entropy(X)

#pickle dump
import bz2
f = bz2.BZ2File(out_pkl, "wb")
pickle.dump(scores, f)
f.close()

# import time
# time.sleep(60)

