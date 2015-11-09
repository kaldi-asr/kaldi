#!/usr/bin/env python
import sys, os, logging, numpy as np
import numpy.matlib
import cPickle as pickle, bz2

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from io_funcs import kaldi_io, utils

from optparse import OptionParser
usage = "%prog [options] <data-dir> <num-strms>"
parser = OptionParser(usage)

parser.add_option('--pvals', dest="pvals",
                  help="prob values for each combination [default: %default]",
                  default="", type='string')

parser.add_option('--frame-level', dest="frame_level",
                  help="Change strm masks for each frame [default: %default]",
                  default="false", type='string')

(o, args) = parser.parse_args()
if len(args) != 2:
  parser.print_help()
  sys.exit(1)

## Create log file
logging.basicConfig(stream=sys.stderr, format='%(asctime)s: %(message)s', level=logging.INFO)

logging.info(" Running as %s ", sys.argv[0])
logging.info(" %s", " ".join(sys.argv))

data_dir=args[0]
nstrms=int(args[1])

ncombs=np.power(2, nstrms)-1

#convert o.pvals
if o.pvals != "":
  pvals = np.asarray( map(lambda x: float(x), o.pvals.split(",")))
  assert ncombs == len(pvals)
else:
  pvals = None

if pvals==None:
  pvals=[1.0/ncombs]*ncombs

data_scp=data_dir+"/feats.scp"
with kaldi_io.KaldiScpReader(data_scp) as data_it:
  for ii, (X, utt) in enumerate(data_it):
    logging.info("processing utt = %s", utt)

    wts = np.zeros((X.shape[0], nstrms))
    
    if o.frame_level == "true":
      comb_idxs=np.random.multinomial(1, pvals, size=X.shape[0])
      for ii, c in enumerate(comb_idxs):
        bin_str='{:010b}'.format(np.nonzero(c)[0][0]+1)
        bin_str=bin_str[-nstrms:]
        wts[ii,:] = np.asarray(map(lambda x: int(x), bin_str))

    else:
      comb_idxs=np.random.multinomial(1, pvals, size=1)      
      c=comb_idxs[0]
      bin_str='{:010b}'.format(np.nonzero(c)[0][0]+1)
      bin_str=bin_str[-nstrms:]

      logging.info(" random comb= %d", np.nonzero(c)[0][0]+1)
      
      wts=wts + np.asarray(map(lambda x: int(x), bin_str))

    Wts = wts
    
    kaldi_io.write_stdout_ascii(Wts, utt)

sys.exit(0)


