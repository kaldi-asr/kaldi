#!/usr/bin/env python

# Copyright 2015  Brno University of Technology (author: Karel Vesely)
# Apache 2.0

import sys, operator

# This scripts loads a 'ctm' file and converts it into the 'tra' format:
# "utt-key word1 word2 word3 ... wordN"
# The 'utt-key' is the 1st column in the CTM.

# Typically the CTM contains:
# - utterance-relative timimng (i.e. prepared without 'utils/convert_ctm.pl')
# - confidences 

if len(sys.argv) != 3:
  print 'Usage: %s ctm-in tra-out' % __file__
  sys.exit(1)
dummy, ctm_in, tra_out = sys.argv

if ctm_in == '-': ctm_in = '/dev/stdin'
if tra_out == '-': tra_out = '/dev/stdout'

# Load the 'ctm' into dictionary,
tra = dict()
with open(ctm_in) as f:
  for l in f:
    utt, ch, beg, dur, wrd, conf = l.split()
    if not utt in tra: tra[utt] = []
    tra[utt].append((float(beg),wrd))

# Store the in 'tra' format,
with open(tra_out,'w') as f:
  for utt,tuples in tra.iteritems():
    tuples.sort(key = operator.itemgetter(0)) # Sort by 'beg' time,
    f.write('%s %s\n' % (utt,' '.join([t[1] for t in tuples])))

