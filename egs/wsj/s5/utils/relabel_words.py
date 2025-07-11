#!/usr/bin/env python3
# Relabel words for lookahead

import sys

lmap = {}
for line in open(sys.argv[1]):
    items = line.split()
    lmap[items[0]] = items[1]

for line in open(sys.argv[2]):
    line = line.strip()
    word, id = line.split()
    if word in set(["<eps>", "<s>", "</s>"]):
        print (line)
    else:
        print (word, lmap[id])
