#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os, sys

# This script is originally from qatip project (http://qatsdemo.cloudapp.net/qatip/demo/)
# of Qatar Computing Research Institute (http://qcri.qa/)

# Remove all phonemes which are not in the phonemeset from extra_question.txt

d = dict()
with open(sys.argv[1]) as f:
    for x in f:
        d[x.strip()] = True

for line in sys.stdin:
    str = ""
    for word in line.strip().split():
        if word in d:
            str = str+" "+word
    sys.stdout.write(str.strip() + "\n")
