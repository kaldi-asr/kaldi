#!/usr/bin/env python

# Copyright 2014  Gaurav Kumar.   Apache 2.0
# Gets the unique speakers from the file created by fsp_make_trans.pl
# Note that if a speaker appears multiple times, it is categorized as female

import os
import sys

tmpFileLocation = 'data/local/tmp/spk2gendertmp'

tmpFile = None

try:
     tmpFile = open(tmpFileLocation)
except IOError:
    print 'The file spk2gendertmp does not exist. Run fsp_make_trans.pl first?'

speakers = {}

for line in tmpFile:
    comp = line.split(' ')
    if comp[0] in speakers:
        speakers[comp[0]] = "f"
    else:
        speakers[comp[0]] = comp[1]

for speaker, gender in speakers.iteritems():
    print speaker + " " + gender
