#!/usr/bin/env python3

import kaldi

f = kaldi.FloatMatrix(2, 3)
f[1, 2] = 100
print(f)

g = f.numpy()
g[0, 0] = 200
print(f)
