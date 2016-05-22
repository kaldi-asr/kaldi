#!/usr/bin/env python
import sys, os, logging, numpy as np
import numpy.matlib
import cPickle as pickle, bz2

from optparse import OptionParser
usage = "%prog [options] scores.1.pklz scores.2.pklz [scores.3.pklz] out_scores.pklz"
parser = OptionParser(usage)

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

out_scores = dict()
for inp_scores_pklz in inp_scores_pklz_list:
  d = pickle.load(bz2.BZ2File(inp_scores_pklz, "rb"))
  out_scores.update(d)

# pickle dump
f = bz2.BZ2File(out_scores_pklz, "wb")
pickle.dump(out_scores, f)
f.close()

sys.exit(0)

