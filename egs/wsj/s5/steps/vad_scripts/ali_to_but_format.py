#!/usr/bin/env python
import sys, os, logging, numpy as np
import numpy.matlib
import cPickle as pickle, bz2
import itertools

from optparse import OptionParser
usage = "%prog [options] vad.ali"
parser = OptionParser(usage)

parser.add_option('--segments', dest="segs_file",
                  help="Kaldi segments file. Alignments are per utt, segments have wav info [default: %default]",
                  default="", type='string')

(o, args) = parser.parse_args()
if len(args) != 1:
  parser.print_help()
  sys.exit(1)

## Create log file
logging.basicConfig(stream=sys.stderr, format='%(asctime)s: %(message)s', level=logging.INFO)

logging.info(" Running as %s ", sys.argv[0])
logging.info(" %s", " ".join(sys.argv))

vad_ali_file=args[0]

f = open(vad_ali_file, "r")
for line in f:
  utt_id = line.split()[0]
  utt_vad_ali = np.asarray(map(lambda x: int(x), line.split()[1:]))
  
  # 
  y=np.diff(utt_vad_ali)
  start_frm_num = np.nonzero(y==1)[0]
  end_frm_num = np.nonzero(y==-1)[0]

  for (st, en) in itertools.izip(start_frm_num, end_frm_num):
    assert st < en
    print ("%s[%d,%d]" %(utt_id, st, en))


sys.exit(0)


