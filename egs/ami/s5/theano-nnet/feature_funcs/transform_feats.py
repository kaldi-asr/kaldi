import sys, os
import numpy as np

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import features
from io_funcs import kaldi_io

class transform_feats:
  
  def __init__(self, transform_feats_file):

    if transform_feats_file == "":
      self.type == "none"

    else:
      import cPickle as pickle, bz2
      d = pickle.load(bz2.BZ2File(transform_feats_file, "rb"))
    
      self.A = d['A']
      self.type = d['type']

  def transform_feats(self, Y, utt):

    if self.type == "linear":
      assert Y.shape[1] == self.A.shape[0]
      return np.dot( Y, self.A)

    elif self.type == "affine":
      logging.info("  Error affine transform not yet implemented")
      assert 1 == 0

    elif self.type == "none":
      return Y


