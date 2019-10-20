This was tested as follows.  After configuring with --shared and making the rest of Kaldi,
you can cd to here and do:
```
# make
# python3
>>> import kaldi_pybind as k
>>> str(k.FloatVector(10))
' [ 0 0 0 0 0 0 0 0 0 0 ]\n'
>>>
```
