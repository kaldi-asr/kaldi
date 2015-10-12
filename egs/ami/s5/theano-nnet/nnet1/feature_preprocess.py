import sys, os
import numpy as np

#import mypytel.features, mypytel.utils
import kaldi_io, features

class FeaturePreprocess:

  def __init__(self, parser_opts):

    (self.glob_mean, self.glob_std) = (0.0, 1.0)

    self.splice_vec = None
    self.set_splice_vec(parser_opts)

    self.delta_scales = None
    self.set_delta_scales(parser_opts)

    self.norm_means = False
    self.norm_vars = False
    if parser_opts.norm_means == "true":
      self.norm_means = True
    if parser_opts.norm_vars == "true":
      self.norm_vars = True

  def set_splice_vec(self, parser_opts):
    if len(parser_opts.splice_vec):
      self.splice_vec = map(lambda x: int(x), parser_opts.splice_vec.split(","))
    else:
      splice = parser_opts.splice
      splice_step = parser_opts.splice_step
      
      self.splice_vec = range(-splice*splice_step, splice*splice_step+1, splice_step)

  def set_delta_scales(self, parser_opts):

    delta_order  = parser_opts.delta_order
    delta_window = parser_opts.delta_window
    
    self.delta_scales = []
    self.delta_scales.append( np.ones(1, dtype='float32') )

    win = np.arange(-delta_window, +(delta_window+1), 1)
    # normalization const
    K = np.arange(1, delta_window+1, dtype='float32')
    K = 2*np.sum(np.power(K, 2))
    win = win/K

    for i in xrange(1, delta_order+1):
      self.delta_scales.append( np.convolve( self.delta_scales[i-1], win) )

  def preprocess(self, fea):
    #return mypytel.features.do_splice(mypytel.features.do_deltas(fea, self.delta_scales), self.splice_vec)
    return self.apply_global_mean_std(self.apply_delta_splice(fea))

  def apply_delta_splice(self, fea):
    return features.do_splice(features.do_deltas(fea, self.delta_scales), self.splice_vec)

  def apply_global_mean_std(self, fea):
    return (fea - self.glob_mean)/self.glob_std

  
def full_preprocess(fea, utt, feat_preprocess, utt2spk_dict, cmvn_dict):
  # spk cmvn
  fea = apply_cmvn(fea, utt, utt2spk_dict, cmvn_dict, feat_preprocess.norm_means, feat_preprocess.norm_vars)
  # delta and context
  return feat_preprocess.preprocess(fea)


class CMVN:
  
  def __init__(self, feat_preprocess, utt2spk_file=None, cmvn_scp=None):
  
    self.utt2spk_dict = {}
    self.cmvn_dict = {}
    
    if feat_preprocess.norm_means == True or feat_preprocess.norm_vars == True:
      if (not os.path.exists(utt2spk_file)) or (not os.path.exists(cmvn_scp)):
        print ("ERROR: utt2spk_file=%s or cmvn_scp=%s, does not exists" % (utt2spk_file, cmvn_scp))
        sys.exit(1)
        

      with open(utt2spk_file) as fd:
        self.utt2spk_dict = dict(line.strip().split(None, 1) for line in fd)

      with kaldi_io.KaldiScpReader(cmvn_scp) as data_it:
        for jj, (X, spk) in enumerate(data_it):
          self.cmvn_dict[spk] = X

def apply_cmvn(fea, utt, utt2spk_dict, cmvn_dict, norm_means=True, norm_vars=True):
  if norm_means == False and norm_vars == False:
    return fea.astype("float32")

  n = cmvn_dict[utt2spk_dict[utt]][0,-1]
  f = cmvn_dict[utt2spk_dict[utt]][0,0:fea.shape[1]].astype("float32")
  s = cmvn_dict[utt2spk_dict[utt]][1,0:fea.shape[1]].astype("float32")

  mean = f / n
  std = np.sqrt(s / n - mean**2)

  if norm_means:
    fea = (fea - mean)
  if norm_vars:
    fea = fea / std

  return fea.astype("float32")

