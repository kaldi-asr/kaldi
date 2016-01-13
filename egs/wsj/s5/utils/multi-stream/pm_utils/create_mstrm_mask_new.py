#!/usr/bin/env python
import sys, os, logging, numpy as np
import numpy.matlib
import cPickle as pickle, bz2

sys.path.append(os.path.join(os.path.dirname(__file__), '../../numpy_io'))
import kaldi_io

from optparse import OptionParser
usage = "%prog [options] <scores-scp>"
parser = OptionParser(usage)

(o, args) = parser.parse_args()
if len(args) != 1:
  parser.print_help()
  sys.exit(1)

## Create log file
logging.basicConfig(stream=sys.stderr, format='%(asctime)s: %(message)s', level=logging.INFO)

logging.info(" Running as %s ", sys.argv[0])
logging.info(" %s", " ".join(sys.argv))

in_ark='/dev/stdin'
out_ark='/dev/stdout'
scores_scp=args[0]

#load the dicts
scores_dicts=[]
f = open(scores_scp,'r')
for line in f.readlines():
  d = pickle.load(bz2.BZ2File(line.strip(), "rb"))
  scores_dicts.append(d)
f.close()

nstrms=int(np.log2(len(scores_dicts)+1))
with open(out_ark, 'wb') as output:
  for ii, (utt, X) in enumerate(kaldi_io.read_mat_ark(in_ark)):
    logging.info("processing utt = %s", utt)
    
    goodness_scores = map(lambda x: x[utt], scores_dicts)
    best_comb = np.argmax(goodness_scores)+1 #[0..N-1]->[1..N]

    #logging.info("  best_comb = %d", best_comb)
 
    #best_comb -> strm_mask    
    bin_str='{:010b}'.format(best_comb)
    bin_str=bin_str[-nstrms:]
    wts = np.asarray(map(lambda x: int(x), bin_str))

    logging.info("  best_comb=%d, %s", best_comb, wts)
    
    Wts = np.matlib.repmat(wts, X.shape[0], 1)

    kaldi_io.write_mat(output, Wts, key=utt)
    
sys.exit(0)


