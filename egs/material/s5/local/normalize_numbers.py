#!/usr/bin/env python
# Converts numbers to their text representations
# Reads from stdin

import os
import sys
__location__ = os.path.realpath(
    os.path.join(os.getcwd(), os.path.dirname(__file__)))

# Read translations of numbers into a dict
num_trans = dict()
with open(sys.argv[1]) as s_f:
  for line in s_f:
    line_comp = line.strip().split('\t')
    num_trans[int(line_comp[0])] = line_comp[1]

# Read input line by line and translate integers
# Will only work for positive integers
# Will not handle numbers which have a comma in them
for line in sys.stdin:
  words = line.strip().split()
  for i in range(len(words)):
    if words[i].isdigit() and int(words[i]) in num_trans:
      words[i] = num_trans[int(words[i])]

  sys.stdout.write(" ".join(words) + "\n")
