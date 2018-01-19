#!/usr/bin/env python3
# Copyright 2017 Bengu Wu
# Apache 2.0.

# This script generate trials file.
# Trial file is formatted as:
# uttid spkid target|nontarget

# If uttid belong to spkid, it is marked 'target',
# otherwise is 'nontarget'.
# input: eval set uttspk file
# output: trial file

import sys

fnutt = sys.argv[1]
ftrial = open(sys.argv[2], 'w')

dictutt = {}
for line in open(fnutt):
  utt2spk = line.rstrip('\r\t\n ')
  spk = utt2spk.split(' ')[1]
  if spk not in dictutt:
    dictutt[spk] = spk

for line in open(fnutt):
  utt2spk = line.rstrip('\r\t\n ')
  utt, spk = utt2spk.split(' ')
  for target in dictutt:
    if target == spk:
      trial = utt + ' ' + target + ' target'
    else:
      trial = utt + ' ' + target + ' nontarget'
    ftrial.write(trial + '\n')
ftrial.close()
