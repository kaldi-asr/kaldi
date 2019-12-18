This was tested as follows.  After configuring with --shared and making the rest of Kaldi,
you can cd to here and do:
```
# make
# python3
>>> import kaldi_pybind as k
>>> import numpy as np
>>> a = k.FloatVector(10)
>>> b = np.array(a, copy = False)
>>> b[5:10] = 2.0
>>> str(a)
[ 0 0 0 0 0 2 2 2 2 2 ]

```
