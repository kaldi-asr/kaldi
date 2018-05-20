# Copyright 2015   David Snyder
# Apache 2.0.
#
# Given a trials and scores file, this script 
# prepares input for the binary compute-eer. 
import sys
trials = open(sys.argv[1], 'r').readlines()
scores = open(sys.argv[2], 'r').readlines()
spkrutt2target = {}
for line in trials:
  spkr, utt, target = line.strip().split()
  spkrutt2target[spkr+utt]=target
for line in scores:
  spkr, utt, score = line.strip().split()
  if spkrutt2target.has_key(spkr+utt):
    print score, spkrutt2target[spkr+utt]
