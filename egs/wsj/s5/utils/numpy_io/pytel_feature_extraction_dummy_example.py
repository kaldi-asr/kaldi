#!/usr/bin/env python

# Copyright 2016  Brno University of Technology (author: Karel Vesely)
# Apache 2.0

import numpy as np
import kaldi_io
import sys

# Tested with this pipeline:
# egs/swbd/s5c$ extract-segments scp:data/eval2000/wav.scp data/eval2000/segments ark:- | \
#   utils/numpy_io/pytel_feature_example.py | copy-feats ark:- ark:foo.ark

def framing(a, window, shift=1):
    shape = ((a.shape[0] - window) / shift + 1, window) + a.shape[1:]
    strides = (a.strides[0]*shift,a.strides[0]) + a.strides[1:]
    return np.lib.stride_tricks.as_strided(a, shape=shape, strides=strides)

for key, waveform, samp_freq in kaldi_io.read_wav_ark('/dev/stdin'):
  # We append a feature with value '0', to do something,
  print >> sys.stderr, key, 'Samp-Freq:', samp_freq, 'N-Samples:', len(waveform)

  # Constants,
  window_size = int(0.025 * samp_freq)
  window_shift = int(0.010 * samp_freq)

  # We do framing (should be extended to do feature extraction),
  feats = framing(waveform, window_size, window_shift)

  # Write to stdout,
  kaldi_io.write_mat('/dev/stdout', feats, key=key)
