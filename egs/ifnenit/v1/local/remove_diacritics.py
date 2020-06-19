#!/usr/bin/env python3

# This script is originally from qatip project (http://qatsdemo.cloudapp.net/qatip/demo/)
# of Qatar Computing Research Institute (http://qcri.qa/)

# Convert unicode transcripts to Normal Form D (NFD).
# Delete Mark,Nonspacing unicode characters.

import unicodedata
import sys, io
def strip_accents(s):
    return ''.join(c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn')

in_stream = io.TextIOWrapper(sys.stdin.buffer, encoding='utf-8')
out_stream = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
for line in in_stream:
    out_stream.write(strip_accents(line))
