#!/usr/bin/env python

# Apache 2.0

import sys

vocab=set()
with open(sys.argv[1]) as vocabfile:
    for line in vocabfile:
        vocab.add(line.strip())

with open(sys.argv[2]) as textfile:
    for line in textfile:
        print " ".join(map(lambda word: word if word in vocab else '<UNK>', line.strip().split()))
