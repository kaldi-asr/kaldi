#!/usr/bin/env python

# Copyright 2019 Johns Hopkins University (Author: Jinyi Yang)

# This script keeps the top three pronunciation probabilities for
# the words with multiple pronunciations. It takes the lexiconp.txt file as input, and
# write the output to stdout.

import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
max_prons = 3
lex = {}

with open(sys.argv[1], 'r', encoding='utf-8') as lexfile:
  for line in lexfile:
    tokens = line.strip().split()
    word = tokens.pop(0)
    #prob = tokens.pop(0)
    if word in lex.keys(): # Found a word with multiple pronunciations
      lex[word].append(tokens)
    else:
      lex[word] = [tokens]

for key, values in lex.items():
  if len(values) > max_prons:
    values_sorted = sorted(values, key=lambda v:v[0], reverse=True)
    values = values_sorted[:max_prons]
  for v in values:
    print(key, " ".join(v))



