#!/usr/bin/env python
# Copyright 2017   Hossein Hadian

# Apache 2.0



import sys
import math

counts = {}
tot_count = 0
for line in sys.stdin:
  words = line.strip().split()
  # tot_count += len(words)
  for w in words:
    if w in counts:
      continue
    else:
      tot_count += 1
      # counts[w] = counts[w] + 1 if w in counts else 1
      counts[w] = 1

print("0\t1\tSIL\tSIL")
for w in counts:
  # logp = -math.log(float(counts[w]) / tot_count)
  logp = -math.log(float(1) / tot_count)
  print("1\t2\t{w}\t{w}\t{p}".format(w=w, p=logp))

print("2")
print("2\t3\tSIL\tSIL")
print("3")
