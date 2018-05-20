#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Copyright 2018  Tsinghua University (Author: Zhiyuan Tang)
# Apache 2.0.

'''
  This script converts nnet3 output like

      utt_id [ 
         0.1  0.2  -0.3  0.5 ...
         0.5  0.8  0.2   0.7 ... ]

  to the form

      utt_id  0.1  0.2  -0.3  0.5 ...
      utt_id  0.5  0.8  0.2   0.7 ...

  i.e., each frame has its utt-id. 
'''

import sys


if len(sys.argv) != 3:
  print 'usage: output_format_frame.py <src> <des>'
  sys.exit()

lines = open(sys.argv[1], 'r').readlines()
des = open(sys.argv[2], 'w')
for line in lines:
  items = line.strip().strip("[]").split()
  if len(items) == 1:
    utt = items[0]
    continue
  des.write(utt + ' ' + ' '.join(items) + '\n')
des.close()

