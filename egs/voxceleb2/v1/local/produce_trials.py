#!/usr/bin/env python3
# Copyright 2017 Bengu Wu
#           2018 Ewald Enzinger
# Apache 2.0.

# Original version copied from egs/aishell/v1/local/produce_trials.py
# (commit 35950ea2461f63e7de9423456c13abb22a396ac7)

# This script generates a trials file.
# Trial file is formatted as:
# uttid uttid target|nontarget

# The utterance ID is used as speaker ID in order to treat
# test utterances of a given speaker independently from each other.

# If uttid belong to spkid, it is marked 'target',
# otherwise is 'nontarget'.
# input: test set uttspk file
# output: trial file

import sys

utt2spk_file = sys.argv[1]
trial_file = open(sys.argv[2], 'w')

utt_dict = {}
for line in open(utt2spk_file):
  utt2spk = line.rstrip('\r\t\n ')
  utt, spk = utt2spk.split(' ')
  if utt not in utt_dict:
    utt_dict[utt] = spk

for line in open(utt2spk_file):
  utt2spk = line.rstrip('\r\t\n ')
  utt, spk = utt2spk.split(' ')
  for target in utt_dict:
    if target != utt:
      if utt_dict[target] == spk:
        trial = utt + ' ' + target + ' target'
      else:
        trial = utt + ' ' + target + ' nontarget'
      trial_file.write(trial + '\n')
  utt_dict.pop(utt)
trial_file.close()
