import sys
import numpy as np

def do_splice(X, splice_vec=[0]):
    
  (num_frames, inp_fea_dim) = X.shape

  out_fea_dim = len(splice_vec) * inp_fea_dim
  Y = np.zeros((num_frames, out_fea_dim), dtype='float32')

  for i in xrange(num_frames):
    
    cur_frm_id = i
    j = 0
    for s in splice_vec:
      f = cur_frm_id + s

      if f < 0:
        f = 0
      if f > num_frames-1:
        f = num_frames-1

      st = j * inp_fea_dim
      en = (j + 1) * inp_fea_dim

      Y[cur_frm_id, st:en] = X[f, :]

      j = j+1

  return Y


def do_deltas(x, delta_scales):

  def process(X, frame):
    (num_frames, fea_dim) = X.shape
    
    Y_frame = np.zeros(fea_dim*( (len(delta_scales)-1) +1), dtype='float32')

    for o in xrange(len(delta_scales)):

      scale = delta_scales[o]
      offset = (len(scale) - 1)/2
       
      for j in xrange(-offset, offset+1, 1):
        c = j + frame
        if c < 0:
          c = 0
        if c >= num_frames:
          c = num_frames - 1

        Y_frame[o*fea_dim:(o+1)*fea_dim] = Y_frame[o*fea_dim:(o+1)*fea_dim] \
                                       + scale[j+offset]*X[c,:]

    return Y_frame

  (num_frames, fea_dim) = x.shape
  y = np.zeros((num_frames, ((len(delta_scales)-1)+1)*fea_dim), dtype='float32')

  for r in xrange(num_frames):
    y[r,:] = process(x, r)

  return y


def fea_lab_pair_iter(data_it, lab_dict, length_tolerance=10):

  for fea, utt in data_it:
    # check if key exists in lab_dict
    if utt in lab_dict.keys():
      targets = lab_dict[utt]
        
      # fix length mismatch
      min_len = min(len(targets), fea.shape[0])
      max_len = max(len(targets), fea.shape[0])
      if (max_len - min_len) < length_tolerance:
        if len(targets) != min_len: targets = targets[0:min_len]
        if fea.shape[0] != min_len: fea = fea[0:min_len,:]
      else:
        print utt + ", length mismatch of targets " + str(len(targets)) + \
            " and features " + str(fea.shape[0])
        continue
        
      yield fea, targets

    else:
      continue

def apply_cmvn(fea, utt, utt2spk_dict, cmvn_dict, norm_means=True, norm_vars=True):
  
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

