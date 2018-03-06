#!/usr/bin/python
# -*- coding: utf-8 -*-
import os, sys

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
