#!/usr/bin/env python
import sys, os, logging, numpy as np
import numpy.matlib
import cPickle as pickle, bz2

from optparse import OptionParser
usage = "%prog [options] scores.inp.pklz scores.out.txt"
parser = OptionParser(usage)

(o, args) = parser.parse_args()
if len(args) != 2:
  parser.print_help()
  sys.exit(1)

## Create log file
logging.basicConfig(stream=sys.stderr, format='%(asctime)s: %(message)s', level=logging.INFO)

logging.info(" Running as %s ", sys.argv[0])
logging.info(" %s", " ".join(sys.argv))

inp_scores_pklz = args[0]
out_scores_txt = args[1]

d = pickle.load(bz2.BZ2File(inp_scores_pklz, "rb"))

fp = open(out_scores_txt, "w")
for k in d.keys():
  fp.write(k+" ")
  for v in d[k]:
    fp.write(str(v)+" ")
  fp.write("\n")
fp.close()

sys.exit(0)

