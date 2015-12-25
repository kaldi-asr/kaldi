#!/usr/bin/env python
import sys, os, logging, numpy as np
import numpy.matlib
import cPickle as pickle, bz2

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from io_funcs import kaldi_io, utils

def scores_to_best_comb(scores_dicts_list, utt, combination_type="sum"):

  #number of methods
  num_pm_methods = len(scores_dicts_list)
  num_combs = len(scores_dicts_list[0])
  goodness_scores = np.asarray((num_pm_methods, num_combs))

  for ii, scores_dicts in enumerate(scores_dicts_list):
    goodness_scores[ii, :] = np.asarray(map(lambda x: x[utt], scores_dicts))
    
  #combine scores from the methods
  if ii == 0: # then only 1 method
    best_comb = np.argmax(goodness_scores)+1 #[0..30]->[1..31]
  elif ii > 0: # multiple methods
    s = 1/np.sum(goodness_scores, axis=1)
    S = np.transpose(np.matlib.repmat(s, goodness_scores.shape[1], 1))
    goodness_scores = goodness_scores * S
    
    if combination_type == "sum": #Arith. Mean of probs
      goodness_scores = np.mean(goodness_scores, axis=0)
      best_comb = np.argmax(goodness_scores)+1 #[0..30]->[1..31]
    elif combination_type == "prod": #Geo Mean of probs
      log_goodness_scores = np.log(goodness_scores)
      mean_log_goodness_scores = np.mean(log_goodness_scores, axis=0)
      goodness_scores = np.exp(mean_log_goodness_scores)
      goodness_scores = goodness_scores/np.sum(goodness_scores)

      best_comb = np.argmax(goodness_scores)+1 #[0..30]->[1..31]

  else:
    logging.info("ERROR: Wrong number of methods. Please check");
    sys.exit(1)

  return best_comb

from optparse import OptionParser
usage = "%prog [options] <data-dir> <scores1.scp> <scores2.scp> ..."
parser = OptionParser(usage)

parser.add_option('--combination-type', dest='combination_type',
                  help='type of combination of performance monitor weights [default: %default]',
                  default='sum', type='str');

(o, args) = parser.parse_args()
if len(args) < 2:
  parser.print_help()
  sys.exit(1)

## Create log file
logging.basicConfig(stream=sys.stderr, format='%(asctime)s: %(message)s', level=logging.INFO)

logging.info(" Running as %s ", sys.argv[0])
logging.info(" %s", " ".join(sys.argv))

data_dir=args[0]
scores_scp_list=args[1:]

#load the dicts
scores_dicts_list=[] #each element in this list correspond to scores from a PM method
for scores_scp in scores_scp_list:
  scores_dicts=[]
  f = open(scores_scp,'r')
  for line in f.readlines():
    d = pickle.load(bz2.BZ2File(line.strip(), "rb"))
    scores_dicts.append(d)
  f.close()

  scores_dicts_list.append(scores_dicts)


nstrms=int(np.log2(len(scores_dicts_list[0])+1))
data_scp=data_dir+"/feats.scp"
with kaldi_io.KaldiScpReader(data_scp) as data_it:
  for ii, (X, utt) in enumerate(data_it):
    logging.info("processing utt = %s", utt)

    best_comb = scores_to_best_comb(scores_dicts_list, utt, o.combination_type)

    #best_comb -> strm_mask    
    bin_str='{:010b}'.format(best_comb)
    bin_str=bin_str[-nstrms:]
    wts = np.asarray(map(lambda x: int(x), bin_str))

    logging.info("  best_comb=%d, %s", best_comb, wts)
    
    Wts = np.matlib.repmat(wts, X.shape[0], 1)
    
    kaldi_io.write_stdout_ascii(Wts, utt)

sys.exit(0)


