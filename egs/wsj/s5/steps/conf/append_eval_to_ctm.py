#!/bin/env python

# Copyright 2015  Brno University of Technology (author: Karel Vesely)
# Apache 2.0

import sys
import numpy as np

# Append Levenshtein alignment of 'hypothesis' and 'reference' into 'CTM':
# (i.e. the output of 'align-text' post-processed by 'wer_per_utt_details.pl')

# The tags in the appended column are:
#  'C' = correct
#  'S' = substitution
#  'I' = insertion
#  'U' = unknown (not part of scored segment)

if len(sys.argv) != 4:
  print 'Usage: %s eval-in ctm-in ctm-eval-out' % __file__
  sys.exit(1)
dummy, eval_in, ctm_in, ctm_eval_out = sys.argv

if ctm_eval_out == '-': ctm_eval_out = '/dev/stdout'

# Read the evalutation,
eval_vec = dict()
with open(eval_in, 'r') as f:
  while True:
    # Reading 4 lines encoding one utterance,
    ref = f.readline()
    hyp = f.readline()
    op = f.readline()
    csid = f.readline()
    if not ref: break
    # Parse the input,
    utt,tag,hyp_vec = hyp.split(' ',2)
    assert(tag == 'hyp')
    utt,tag,op_vec = op.split(' ',2)
    assert(tag == 'op')
    eval_vec[utt] = np.array(op_vec.split())[np.array(hyp_vec.split()) != '<eps>']

# Read the CTM (contains confidences),
ctm = np.loadtxt(ctm_in, dtype='object,object,f8,f8,object,f8')
ctm = np.sort(ctm, order=['f0','f1','f2'])
# Split CTM per keys from 1st column,
ctm_parts = np.split(ctm, np.nonzero(ctm['f0'][1:] != ctm['f0'][:-1])[0]+1)

# Build the 'ctm' with 'eval' column added,
ctm_eval = []
for part in ctm_parts:
  utt = part[0][0]
  # extending the 'tuple' by '+':
  merged = [ tuple(tup) + (evl,) for tup,evl in zip(part,eval_vec[utt]) ]
  ctm_eval.append(merged)
  
# Store,
import operator
ctm_eval = reduce(operator.add, ctm_eval) # Flattening the array of arrays,
ctm_eval = np.array(ctm_eval, dtype='object,object,f8,f8,object,f8,object')
np.savetxt(ctm_eval_out, ctm_eval, fmt=['%s','%s','%f','%f','%s','%f','%s'])

