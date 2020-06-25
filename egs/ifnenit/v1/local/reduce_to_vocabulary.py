#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os, sys, io

# This script is originally from qatip project (http://qatsdemo.cloudapp.net/qatip/demo/)
# of Qatar Computing Research Institute (http://qcri.qa/)

# Remove all phonemes which are not in the phonemeset from extra_question.txt

d = dict()
with open(sys.argv[1], encoding="utf-8") as f:
    for x in f:
        d[x.strip()] = True

in_stream = io.TextIOWrapper(sys.stdin.buffer, encoding='utf-8')
out_stream = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
for line in in_stream:
    str = ""
    for word in line.strip().split():
        if word in d:
            str = str+" "+word
    out_stream.write(str.strip() + "\n")
