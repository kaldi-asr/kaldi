#!/usr/bin/env python
import sys, os, logging, numpy as np
import cPickle as pickle
import bz2

from optparse import OptionParser
usage = "%prog [options] <inp_dict.1.pkl> <inp_dict.2.pkl> [<inp_dict.N.pkl>] <out_dict.pkl>"
parser = OptionParser(usage)


(o, args) = parser.parse_args()
if len(args) < 3:
  parser.print_help()
  sys.exit(1)

out_pkl_file = args.pop()
inp_pkl_file_list = args

## Create log file
logging.basicConfig(stream=sys.stderr, format='%(asctime)s: %(message)s', level=logging.INFO)

logging.info("  Combining dicts from %s", ' '.join(inp_pkl_file_list))
logging.info("  to -> %s", out_pkl_file)

#Load input dicts
inp_d_list=[]
for inp_pkl in inp_pkl_file_list:
  d = pickle.load(bz2.BZ2File(inp_pkl, "rb"))
  inp_d_list.append(d)


#Combine them 
out_d = {}
for d in inp_d_list:
  out_d.update(d)

#Write, pickle dump
f = bz2.BZ2File(out_pkl_file, "wb")
pickle.dump(out_d, f)
f.close()

sys.exit(0)

