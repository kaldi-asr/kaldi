#!/usr/bin/env python
#
# Copyright  2014  Nickolay V. Shmyrev
#            2016  Johns Hopkins University (author: Daniel Povey)
# Apache 2.0
#
# Based on egs/tedlium/s5_r2/local/join_suffix.py (commit 7a47a1154550861dd7d51af9e249eeb292a1de40).

from __future__ import print_function
import sys
from codecs import open

# This script joins together pairs of split-up words like "you 're" -> "you're".
# The TEDLIUM transcripts are normalized in a way that's not traditional for
# speech recognition.

for line in sys.stdin:
    items = line.split()
    new_items = []
    i = 1
    while i < len(items):
        if i < len(items) - 1 and items[i+1][0] == '\'':
            new_items.append(items[i] + items[i+1])
            i = i + 1
        else:
            new_items.append(items[i])
        i = i + 1
    print(items[0] + ' ' + ' '.join(new_items))
