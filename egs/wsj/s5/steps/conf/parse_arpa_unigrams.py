#!/bin/env python

# Copyright 2015  Brno University of Technology (author: Karel Vesely)
# Apache 2.0

import sys, gzip, re

# Parse options,
if len(sys.argv) != 3:
  print "Usage: %s <arpa-gz> <unigrams>" % __file__
  sys.exit(0)
arpa_gz, unigrams_out = sys.argv[1:]

if arpa_gz == '-': arpa_gz = '/dev/stdin'
if unigrams_out == '-': unigrams_out = '/dev/stdout'

# Load the unigram probabilities in 10log from ARPA,
wrd_log10 = []
with gzip.open(arpa_gz,'r') as f:
  read = False
  for l in f:
    if l.strip() == '\\1-grams:': read = True
    if l.strip() == '\\2-grams:': break
    if read and len(l.split())>=2:
      log10_p_unigram, wrd = re.split('[\t ]+',l.strip(),2)[:2]
      wrd_log10.append((wrd, float(log10_p_unigram)))

# Sort by unigram probability (descending),  
wrd_log10.sort(key=lambda tup: tup[1], reverse=True)

# Store,
with open(unigrams_out,'w') as f:
  f.writelines(['%s %g\n' % pair for pair in wrd_log10])
