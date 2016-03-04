#!/usr/bin/env python

# Copyright 2015  Brno University of Technology (author: Karel Vesely)
# Apache 2.0

import sys

# Append Levenshtein alignment of 'hypothesis' and 'reference' into 'CTM':
# (parsed from the 'prf' output of 'sclite')

# The tags in appended column are:
#  'C' = correct
#  'S' = substitution
#  'I' = insertion
#  'U' = unknown (not part of scored segment)

# Parse options,
if len(sys.argv) != 4:
  print "Usage: %s prf ctm_in ctm_out" % __file__
  sys.exit(1)
prf_file, ctm_file, ctm_out_file = sys.argv[1:]

if ctm_out_file == '-': ctm_out_file = '/dev/stdout'

# Load the prf file,
prf = []
with open(prf_file) as f:
  for l in f:
    # Store the data,
    if l[:5] == 'File:':
      file_id = l.split()[1]
    if l[:8] == 'Channel:':
      chan = l.split()[1]
    if l[:5] == 'H_T1:':
      h_t1 = l
    if l[:5] == 'Eval:':
      evl = l
      prf.append((file_id,chan,h_t1,evl))

# Parse the prf records into dictionary,
prf_dict = dict()
for (f,c,t,e) in prf:
  t_pos = 0 # position in the 't' string,
  while t_pos < len(t):
    t1 = t[t_pos:].split(' ',1)[0] # get 1st token at 't_pos'
    try:
      # get word evaluation letter 'C,S,I',
      evl = e[t_pos] if e[t_pos] != ' ' else 'C' 
      # add to dictionary,
      key='%s,%s' % (f,c) # file,channel
      if key not in prf_dict: prf_dict[key] = dict()
      prf_dict[key][float(t1)] = evl
    except ValueError:
      pass
    t_pos += len(t1)+1 # advance position for parsing,

# Load the ctm file (with confidences),
with open(ctm_file) as f:
  ctm = [ l.split() for l in f ]

# Append the sclite alignment tags to ctm,
ctm_out = []
for f, chan, beg, dur, wrd, conf in ctm:
  # U = unknown, C = correct, S = substitution, I = insertion,
  sclite_tag = 'U' 
  try:
    sclite_tag = prf_dict[('%s,%s'%(f,chan)).lower()][float(beg)]
  except KeyError:
    pass
  ctm_out.append([f,chan,beg,dur,wrd,conf,sclite_tag])

# Save the augmented ctm file,
with open(ctm_out_file, 'w') as f:
  f.writelines([' '.join(ctm_record)+'\n' for ctm_record in ctm_out])

