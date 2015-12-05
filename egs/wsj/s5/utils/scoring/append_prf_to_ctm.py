#!/bin/env python

# Copyright 2015  Brno University of Technology (author: Karel Vesely)
# Apache 2.0

import sys
import numpy as np

# Append sclite alignment-tags to the ctm as last column. 
#
# The tags are:
#  'C' = correct
#  'S' = substitution
#  'I' = insertion
#  'U' = unknown (not part of scored segment)

# Parse options,
if len(sys.argv) != 4:
  print "Usage: %s prf ctm_in ctm_out" % __file__
  sys.exit(1)
dummy, prf_file, ctm_file, ctm_out_file = sys.argv

# Load the prf file,
prf = []
with open(prf_file) as f:
  for l in f:
    # Store the data,
    if l[:5] == 'File:':
      file_id = l.split()[1]
    if l[:5] == 'H_T1:':
      h_t1 = l
    if l[:5] == 'Eval:':
      evl = l
      prf.append((file_id,h_t1,evl))

# Parse the prf records into dictionary,
prf_dict = dict()
for (f,t,e) in prf:
  t_pos = 0
  while t_pos < len(t):
    t1 = t[t_pos:].split(' ',1)[0]
    try:
      t1f = float(t1)
      evl = e[t_pos] if e[t_pos] != ' ' else 'C'
      # add to dictionary,
      if f not in prf_dict: prf_dict[f] = dict()
      prf_dict[f][t1f] = evl
    except ValueError:
      pass
    t_pos += len(t1)+1

# Load the ctm file,
ctm = np.loadtxt(ctm_file, dtype='object,object,f8,f8,object,f8')

# Append the sclite alignment tags to ctm,
ctm2 = []
for (f, chan, beg, dur, wrd, conf) in ctm:
  # U = unknown, C = correct, S = substitution, I = insertion,
  sclite_tag = 'U' 
  try:
    sclite_tag = prf_dict[f.lower()][beg]
  except KeyError:
    pass
  ctm2.append((f,chan,beg,dur,wrd,conf,sclite_tag))
ctm3 = np.array(ctm2, dtype='object,object,f8,f8,object,f8,object')

# Save the augmented ctm file,
np.savetxt(ctm_out_file, ctm3, fmt='%s %s %.2f %.2f %s %.2f %s')

