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

fnutt = sys.argv[1]
ftrial = open(sys.argv[2], 'w')

dictutt = {}
for line in open(fnutt):
  utt2spk = line.rstrip('\r\t\n ')
  utt, spk = utt2spk.split(' ')
  if utt not in dictutt:
    dictutt[utt] = spk

for line in open(fnutt):
  utt2spk = line.rstrip('\r\t\n ')
  utt, spk = utt2spk.split(' ')
  for target in dictutt:
    if target != utt:
      if dictutt[target] == spk:
        trial = utt + ' ' + target + ' target'
      else:
        trial = utt + ' ' + target + ' nontarget'
      ftrial.write(trial + '\n')
  dictutt.pop(utt)
ftrial.close()
