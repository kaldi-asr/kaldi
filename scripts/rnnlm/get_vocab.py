#!/usr/bin/env python3

# Copyright  2017  Jian Wang
# License: Apache 2.0.

import os
import argparse
import sys
sys.stdout = open(1, 'w', encoding='utf-8', closefd=False)

import re


parser = argparse.ArgumentParser(description="This script get a vocab from unigram counts "
                                 "of words produced by get_unigram_counts.sh",
                                 epilog="E.g. " + sys.argv[0] + " data/rnnlm/data > data/rnnlm/vocab/words.txt",
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument("data_dir",
                    help="Directory in which to look for unigram counts.")

args = parser.parse_args()

eos_symbol = '</s>'
special_symbols = ['<s>', '<brk>', '<eps>']


# Add the count for every word in counts_file
# the result is written into word_counts
def add_counts(word_counts, counts_file):
    with open(counts_file, 'r', encoding="utf-8") as f:
        for line in f:
            line = line.strip(" \t\r\n")
            word_and_count = line.split()
            assert len(word_and_count) == 2
            if word_and_count[0] in word_counts:
                word_counts[word_and_count[0]] += int(word_and_count[1])
            else:
                word_counts[word_and_count[0]] = int(word_and_count[1])

word_counts = {}

for f in os.listdir(args.data_dir):
    full_path = args.data_dir + "/" + f
    if os.path.isdir(full_path):
        continue
    if f.endswith(".counts"):
        add_counts(word_counts, full_path)

if len(word_counts) == 0:
    sys.exit(sys.argv[0] + ": Directory {0} should contain at least one .counts file "
             .format(args.data_dir))

print("<eps> 0")
print("<s> 1")
print("</s> 2")
print("<brk> 3")

idx = 4
for word, _ in sorted(word_counts.items(), key=lambda x: x[1], reverse=True):
    if word == "</s>":
        continue
    print("{0} {1}".format(word, idx))
    idx += 1

print(sys.argv[0] + ": vocab is generated with {0} words.".format(idx), file=sys.stderr)
