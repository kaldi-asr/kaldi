#!/usr/bin/env python
import sys, os, logging, numpy as np
import numpy.matlib
import cPickle as pickle, bz2

sys.path.append(os.path.join(os.path.dirname(__file__), '../../numpy_io'))
import kaldi_io

from optparse import OptionParser
usage = "%prog [options] pm1_scores.pklz pm2_scores.pklz [pm3_scores.pklz] out_scores.pklz"
parser = OptionParser(usage)

parser.add_option('--weights', dest='weights',
                  help='weights, seperated by :, to do convex combination of scores [default: %default]',
                  default='', type='string')

parser.add_option('--pm-stats-files', dest='pm_stats_files',
                  help='files, seperated by :, to do mean and variance normalization of perf monitor scores [default: %default]',
                  default='', type='string')


(o, args) = parser.parse_args()
if len(args) < 3:
  parser.print_help()
  sys.exit(1)

## Create log file
logging.basicConfig(stream=sys.stderr, format='%(asctime)s: %(message)s', level=logging.INFO)

logging.info(" Running as %s ", sys.argv[0])
logging.info(" %s", " ".join(sys.argv))

out_scores_pklz = args.pop()
inp_scores_pklz_list = args

inp_scores_list = []
for inp_scores_pklz in inp_scores_pklz_list:
  d = pickle.load(bz2.BZ2File(inp_scores_pklz, "rb"))
  inp_scores_list.append(d)

# load weights
if o.weights == '':
  weights = np.asarray([1/float(len(inp_scores_list))] * len(inp_scores_list))
else:
  weights = np.asarray( map(lambda x: float(x), o.weights.split(':')) )
  # renorming
  weights = weights/np.sum(weights)
weights = weights.tolist()

# get train stats
if o.pm_stats_files == '':
  num_pms = len(inp_scores_list)
  (pm_scores_mus, pm_scores_stds) = ([0.0]*num_pms, [1.0]*num_pms)
else:
  (pm_scores_mus, pm_scores_stds) = ([], [])
  pm_stats_files_list = map(lambda x: x, o.pm_stats_files.split(':'))

  for pm_stats_file in pm_stats_files_list:
    d = pickle.load(bz2.BZ2File(pm_stats_file, "rb"))
    
    this_mu = np.mean(np.asarray(d.values()))
    this_std = np.std(np.asarray(d.values()))

    pm_scores_mus.append(this_mu)
    pm_scores_stds.append(this_std)

    
out_scores = dict()
for utt in inp_scores_list[0].keys():
  out_scores[utt] = 0.0
  for ii, inp_scores in enumerate(inp_scores_list):
    score = inp_scores[utt]
    norm_score = (score - pm_scores_mus[ii])/pm_scores_stds[ii]

    out_scores[utt] += (weights[ii] * norm_score)

  out_scores[utt] = out_scores[utt][0]

# pickle dump
f = bz2.BZ2File(out_scores_pklz, "wb")
pickle.dump(out_scores, f)
f.close()

sys.exit(0)

