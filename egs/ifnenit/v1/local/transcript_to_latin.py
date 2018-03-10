#!/usr/bin/python
# -*- coding: utf-8 -*-

# This script is originally from qatip project (http://qatsdemo.cloudapp.net/qatip/demo/)
# of Qatar Computing Research Institute (http://qcri.qa/)

# Convert every utterance transcript to position dependent latin format using "data/train/words2latin" as dictionary.

import os, sys, re

with open(sys.argv[1]) as f:
    d = dict(x.rstrip().split(None, 1) for x in f)

for line in sys.stdin:
    mappedWords = []
    for word in line.split():
        mappedWords.append(d[word])
    sys.stdout.write(re.sub(" +", " ", "    ~A ".join(mappedWords).strip()) + "\n")
