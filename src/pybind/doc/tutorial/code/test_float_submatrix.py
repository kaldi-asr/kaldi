#!/usr/bin/env python3

import kaldi
import numpy as np

m = np.array([[1, 2, 3], [10, 20, 30]], dtype=np.float32)
f = kaldi.FloatSubMatrix(m)

f[1, 2] = 100
print(m)
print()

g = f.numpy()
g[0, 0] = 200
print(m)
