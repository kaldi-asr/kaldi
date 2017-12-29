#!/usr/bin/env python

# Copyright 2017 Bengu Wu
# Apache 2.0.

import sys,random

dictutt = {}

for eachline in open(sys.argv[1]):
    line = eachline.rstrip('\r\t\n ')
    utt,spk = line.split(' ')
    if spk not in dictutt:
        dictutt[spk] = []
        dictutt[spk].append(utt)
    else:
        dictutt[spk].append(utt)

enrollfile = open(sys.argv[2], 'w')
matchfile = open(sys.argv[3], 'w')

for key in dictutt:
    listutt = dictutt[key]
    random.shuffle(listutt)
    for i in range(0, len(listutt)):
        if(i < 3):
            enrollstr = listutt[i] + ' ' + key
            enrollfile.write(enrollstr + '\n')
        else:
            matchstr = listutt[i] + ' ' + key
            matchfile.write(matchstr + '\n')

enrollfile.close()
matchfile.close()
