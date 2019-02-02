#!/usr/bin/env python3
# This script lexicon file and calculates the OOV rate.

import os
import sys, io

vocab_file = os.path.join('data/local/dict/lexicon.txt')
infile = io.TextIOWrapper(sys.stdin.buffer, encoding='utf-8')
word_dict = dict()
with open(vocab_file, 'r', encoding='utf-8') as vocab_fh:
    for line in vocab_fh:
        line = line.strip().split()[0]
        word_dict[line] = line

word_dict[' '] = ' '
oov_words=0
tot_words=0
for line in infile:
    text = line.strip().split()
    for word in text:
        tot_words += 1
        if word not in word_dict.keys():
            oov_words += 1

print("{} {}".format(tot_words, oov_words))
