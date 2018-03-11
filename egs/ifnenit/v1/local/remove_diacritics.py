#!/usr/bin/env python3

# This script is originally from qatip project (http://qatsdemo.cloudapp.net/qatip/demo/)
# of Qatar Computing Research Institute (http://qcri.qa/)

# Convert unicode transcripts to Normal Form D (NFD).
# Delete Mark,Nonspacing unicode characters.

import unicodedata
import sys

def strip_accents(s):
    return ''.join(c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn')

for line in sys.stdin:
    sys.stdout.write(strip_accents(line))
