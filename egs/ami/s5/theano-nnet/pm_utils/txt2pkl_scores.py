#!/usr/bin/env python
import sys, os, logging, numpy as np
import cPickle as pickle

from optparse import OptionParser
usage = "%prog [options] <scores.txt> <scores.pkl>"
parser = OptionParser(usage)

(o, args) = parser.parse_args()
# options specified in config overides command line

if len(args) != 2:
  parser.print_help()
  sys.exit(1)

(scores_txt, scores_pkl) = (args[0], args[1])

scores={}
f = open(scores_txt, 'r')
for line in f.xreadlines():
  (utt, scr) = (line.strip().split()[0], float(line.strip().split()[1]))
  scores[utt] = scr
  
f.close()

#pickle dump
import bz2
f = bz2.BZ2File(scores_pkl, "wb")
pickle.dump(scores, f)
f.close()


