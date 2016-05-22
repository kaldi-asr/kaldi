#!/usr/bin/env python

import numpy as np
import kaldi_io

for key,matrix in kaldi_io.read_mat_ark('/dev/stdin'):
  # 1st dim = time axis,
  # 2nd dim = feature axis,
  
  # We append a feature with value '0', to do something,
  rows, cols = matrix.shape
  matrix2 = np.zeros((rows,cols+1))
  matrix2[:,:-1] = matrix

  # Write to stdout,
  kaldi_io.write_mat('/dev/stdout', matrix2, key=key)
