#!/usr/bin/env python
import sys, os, logging, numpy as np
import cPickle as pickle
import compute_mtd

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from io_funcs import kaldi_io, utils

import mkl
mkl.set_num_threads(1)
import numexpr
numexpr.set_num_threads(1)

np.random.seed(42)
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
usage = "%prog [options] <post-scp> <mdelta-prior-file> <out.pkl>"
parser = OptionParser(usage)

parser.add_option('--config', dest="config",
                  help="Configuration file to read (this option may be repeated) [default: %default]",
                  default="", type='string')

(o, args) = parser.parse_args()

if len(args) != 3:
  parser.print_help()
  sys.exit(1)

(post_scp, mdelta_prior_file, out_pkl) = (args[0], args[1], args[2])

## Create log file
logging.basicConfig(stream=sys.stderr, format='%(asctime)s: %(message)s', level=logging.INFO)

logging.info(" Running as %s ", sys.argv[0])
logging.info(" %s", " ".join(sys.argv))

pri = np.loadtxt(mdelta_prior_file)
scores={}
with kaldi_io.KaldiScpReader(post_scp) as data_it:
  for ii, (X, utt) in enumerate(data_it): #X is posteriorgram
    logging.info("processing utt = %s, ii= %d", utt, ii)
    scores[utt] = wrap_compute_mdelta(X, pri)

#pickle dump
import bz2
f = bz2.BZ2File(out_pkl, "wb")
pickle.dump(scores, f)
f.close()

# import time
# time.sleep(60)

