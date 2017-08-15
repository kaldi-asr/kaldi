#!/usr/bin/env python
# Copyright 2017   Hossein Hadian

# Apache 2.0



import sys
import math

counts = {}
tot_count = 0
for line in sys.stdin:
  words = line.strip().split()
  tot_count += len(words)
  for w in words:
    counts[w] = counts[w] + 1 if w in counts else 1

for w in counts:
  logp = -math.log(float(counts[w]) / tot_count)
  print("0\t1\t{w}\t{w}\t{p}".format(w=w, p=logp))

print("1")
