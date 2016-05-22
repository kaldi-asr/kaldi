#!/usr/bin/env python
import sys, os, logging, numpy as np
import numpy.matlib
import cPickle as pickle, bz2

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from io_funcs import kaldi_io, utils

from optparse import OptionParser
usage = "%prog [options] <data-dir> <scores1.pkl> [<scores2.pkl> scores3.pkl .. scoresN.pkl]"
parser = OptionParser(usage)

parser.add_option('--normalize', dest="normalize",
                  help="Normalize scores to make sum()=1 [default: %default]",
                  default="true", type='string')

parser.add_option('--max', dest="max",
                  help="Make max=1 and rest=0 [default: %default]",
                  default="true", type='string')

(o, args) = parser.parse_args()
if len(args) < 2:
  parser.print_help()
  sys.exit(1)

## Create log file
logging.basicConfig(stream=sys.stderr, format='%(asctime)s: %(message)s', level=logging.INFO)

logging.info(" Running as %s ", sys.argv[0])
logging.info(" %s", " ".join(sys.argv))

data_dir=args[0]
scores_pkl_list = args[1:]

#load the dicts
scores_dicts=[]
for pkl_file in scores_pkl_list:
  d = pickle.load(bz2.BZ2File(pkl_file, "rb"))
  scores_dicts.append(d)

data_scp=data_dir+"/feats.scp"
with kaldi_io.KaldiScpReader(data_scp) as data_it:
  for ii, (X, utt) in enumerate(data_it):
    logging.info("processing utt = %s", utt)
    
    goodness_scores = map(lambda x: x[utt], scores_dicts)
    
    #renormalize
    wts = np.asarray(goodness_scores)
    wts = np.reshape(wts, len(wts))
    if o.normalize == "true":
      wts = wts/np.sum(wts)

    if o.max == "true":
      wts1 = np.zeros_like(wts)
      wts1[np.argmax(wts)] = 1.0
      wts = wts1

    #reshape
    Wts = np.zeros((X.shape[0], len(wts))) + wts
    
    kaldi_io.write_stdout_ascii(Wts, utt)

      
sys.exit(0)


