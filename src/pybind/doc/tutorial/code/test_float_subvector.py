#!/usr/bin/env python3

import kaldi
import numpy as np

v = np.array([10, 20, 30], dtype=np.float32)
f = kaldi.FloatSubVector(v)

f[0] = 0
print(v)

g = f.numpy()
g[1] = 100
print(v)
