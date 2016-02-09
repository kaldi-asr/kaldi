#!/usr/bin/env python
import sys, os, logging, numpy as np
import numpy.matlib
import cPickle as pickle, bz2

from optparse import OptionParser
usage = """%prog [options] <ctm-file> <utt-file> <scores.pkl>
           utt file is need because of the following reason:
             sometimes we get confidence per-word=-nan(TODO: Check ) 
             and we do not get scores for that utterance. utt-file 
             utt-file defaults all scores -inf"""

parser = OptionParser(usage)

(o, args) = parser.parse_args()
if len(args) != 3:
  parser.print_help()
  sys.exit(1)

## Create log file
logging.basicConfig(stream=sys.stderr, format='%(asctime)s: %(message)s', level=logging.INFO)

logging.info(" Running as %s ", sys.argv[0])
logging.info(" %s", " ".join(sys.argv))

(ctm_file, utt_file, scores_pkl) = (args[0], args[1], args[2])

ctm_confd_dict = {}
for ii, line in enumerate(open(ctm_file).readlines()):
  utt = line.split()[0]
  cfd = float(line.split()[5])
  if utt in ctm_confd_dict.keys():
    ctm_confd_dict[utt].append(cfd) 
  else:
    ctm_confd_dict[utt]=[cfd]

scores={}
# initialize scores
for ii, line in enumerate(open(utt_file, "r").readlines()):
  utt = line.split()[0]
  scores[utt] = -np.inf
  
# take log average
for ii, utt in enumerate(ctm_confd_dict.keys()):
  log_avg_cfd = 0.0
  n_words=0
  for jj, cfd in enumerate(ctm_confd_dict[utt]):
    log_avg_cfd = log_avg_cfd + np.log(cfd)
    n_words = n_words+1
  scores[utt] = log_avg_cfd/n_words

for ii, utt in enumerate(scores.keys()):
  if np.isinf(scores[utt]):
    logging.info("Utt with -inf confd=%s", utt)

logging.info("Num utt in ctm_file = %d", len(scores.keys()))

#pickle dump
import bz2
f = bz2.BZ2File(scores_pkl, "wb")
pickle.dump(scores, f)
f.close()

sys.exit(0)

