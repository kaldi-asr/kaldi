#!/usr/bin/env python

###########################################################################################
# This script was copied from egs/tedlium/s5/local/join_suffix.py
# The source commit was e69198c3dc5633f98eb88e1cdf20b2521a598f21
# No changes were made
###########################################################################################

# Copyright  2014 Nickolay V. Shmyrev 
# Apache 2.0


import sys
from codecs import open

words = set()
for line in open(sys.argv[1], encoding='utf8'):
    items = line.split()
    words.add(items[0])

for line in sys.stdin:
    items = line.split()
    new_items = []
    i = 1
    while i < len(items):
        if i < len(items) - 1 and items[i+1][0] == '\'' and items[i] + items[i+1] in words:
            new_items.append(items[i] + items[i+1])
            i = i + 1
        else:
            new_items.append(items[i])
        i = i + 1
    print(items[0] + ' ' + ' '.join(new_items))
