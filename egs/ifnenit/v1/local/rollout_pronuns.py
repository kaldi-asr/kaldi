#!/usr/bin/python
# -*- coding: utf-8 -*-
import os, sys, re

with open(sys.argv[1]) as f:
    d = dict(x.rstrip().split(None, 1) for x in f)

for line in sys.stdin:
    mappedWords = []
    for word in line.split():
        mappedWords.append(d[word])
    sys.stdout.write(re.sub(" +", " ", "    ~A ".join(mappedWords).strip()) + "\n")
