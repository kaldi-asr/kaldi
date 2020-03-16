#!/usr/bin/env python3

import kaldi

m = kaldi.FloatMatrix(2, 2)
m[0, 0] = 10
m[1, 1] = 20

xfilename = '/tmp/lda.mat'
kaldi.write_mat(m, xfilename, binary=True)

g = kaldi.read_mat(xfilename)
print(g)
