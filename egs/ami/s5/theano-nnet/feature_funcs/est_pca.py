#!/usr/bin/env python
import sys, os, logging, numpy as np
import numpy.matlib
import cPickle as pickle, bz2

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from io_funcs import kaldi_io, utils

from optparse import OptionParser
usage = "%prog [options] <pca-matrix-out> <pca-acc-1> <pca-acc-2> ... "
parser = OptionParser(usage)

parser.add_option('--dim', dest="dim",
                  help="Feature dimension requested (if <= 0, uses full feature dimension [default: %default]",
                  default="-1", type='string');


(o, args) = parser.parse_args()
if len(args) < 2:
  parser.print_help()
  sys.exit(1)

## Create log file
logging.basicConfig(stream=sys.stderr, format='%(asctime)s: %(message)s', level=logging.INFO)

logging.info(" Running as %s ", sys.argv[0])
logging.info(" %s", " ".join(sys.argv))

pca_transform_file = args[0]
pca_acc_list = args[1:]

#args
dim = int(o.dim)

for ii, pca_acc in enumerate(pca_acc_list):
  logging.info("  loding from %s", pca_acc)
  d = pickle.load(bz2.BZ2File(pca_acc, "rb"))
  if ii == 0:
    N = d['N']
    F = d['F']
    S = d['S']
  else:
    N = N + d['N']
    F = F + d['F']
    S = S + d['S']


# Do EigenAnalysis
sum_mat = F / float(N)
sumsq_mat = S / float(N)
sumsq_mat = sumsq_mat - np.outer(sum_mat, sum_mat)

# Eigen analysis
(w, v) = np.linalg.eigh(sumsq_mat)

w = np.flipud(w)
v = np.fliplr(v)

logging.info("Eigenvalues in PCA are  [ %s ]", map(lambda x: float(x), w))

if dim > 0:
  W = w[0:dim]
  V = v[:,0:dim]
else:
  W = w
  V = v

logging.info("Sum of PCA eigenvalues is %f, sum of kept eigenvalues is %f", sum(w), sum(W))

# Save transform in V
pca_out = {}
pca_out['type'] = "linear"
pca_out['A'] = V

f = bz2.BZ2File(pca_transform_file, "wb")
pickle.dump(pca_out, f)
f.close()

sys.exit(0)


