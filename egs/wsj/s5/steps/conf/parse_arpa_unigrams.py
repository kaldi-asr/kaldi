#!/usr/bin/env python

# Copyright 2015  Brno University of Technology (author: Karel Vesely)
# Apache 2.0

from __future__ import print_function
import sys, gzip, re

# Parse options,
if len(sys.argv) != 4:
  print("Usage: %s <words.txt> <arpa-gz> <unigrams>" % __file__)
  sys.exit(0)
words_txt, arpa_gz, unigrams_out = sys.argv[1:]

if arpa_gz == '-': arpa_gz = '/dev/stdin'
if unigrams_out == '-': unigrams_out = '/dev/stdout'

# Load the words.txt,
words = [ l.split() for l in open(words_txt) ]

# Load the unigram probabilities in 10log from ARPA,
wrd_log10 = dict()
with gzip.open(arpa_gz,'r') as f:
  read = False
  for l in f:
    if l.strip() == '\\1-grams:': read = True
    if l.strip() == '\\2-grams:': break
    if read and len(l.split())>=2:
      log10_p_unigram, wrd = re.split('[\t ]+',l.strip(),2)[:2]
      wrd_log10[wrd] = float(log10_p_unigram)

# Create list, 'wrd id log_p_unigram',
words_unigram = [[wrd, id, (wrd_log10[wrd] if wrd in wrd_log10 else -99)] for wrd,id in words ]

print(words_unigram[0], file=sys.stderr)
# Store,
with open(unigrams_out,'w') as f:
  f.writelines(['%s %s %g\n' % (w,i,p) for (w,i,p) in words_unigram])

