#!/usr/bin/env python3

import kaldi

f = kaldi.FloatVector(3)
f[0] = 10
print(f)

g = f.numpy()
g[1] = 20
print(f)
