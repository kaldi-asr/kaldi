#!/usr/bin/env python2.7
#coding:utf-8
# this script is copied from hkust/s5/local/hkust_segment.py


from __future__ import print_function
import sys
from mmseg import seg_txt
for line in sys.stdin:
  blks = str.split(line)
  out_line = blks[0]
  for i in range(1, len(blks)):
    for j in seg_txt(blks[i]):
      out_line += " " + j
  print(out_line)
