#!/usr/bin/env python

# Copyright 2017 Bengu Wu
# Apache 2.0.

import sys

dictutt = {}
for eachline in open(sys.argv[1]):
    line = eachline.rstrip('\r\t\n ')
    spk = line.split(' ')[1]
    dictutt[spk] = spk

trailfile = open(sys.argv[3], 'w')
for each2 in open(sys.argv[2]):
    line2 = each2.rstrip('\r\t\n ')
    utt2, spk2 = line2.split(' ')
    for spk3 in dictutt:
        if spk3 == spk2:
            trial = utt2 + ' ' + spk3 + ' ' + 'target'
        else:
            trial = utt2 + ' ' + spk3 + ' ' + 'nontarget'
        trailfile.write(trial+'\n')
