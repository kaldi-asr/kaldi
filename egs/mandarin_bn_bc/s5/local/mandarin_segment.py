#!/usr/bin/env python
#coding:utf-8
#!/usr/bin/env python
from __future__ import print_function
import sys
from mmseg import seg_txt
for line in sys.stdin:
  blks = str.split(line)
  out_line = ""
  for i in range(0, len(blks)):
    if blks[i] == "[VOCALIZED-NOISE]" or blks[i] == "[NOISE]" or blks[i] == "[LAUGHTER]":
      out_line += " " + blks[i]
      continue
    for j in seg_txt(blks[i]):
      out_line += " " + j
  print(out_line) 
