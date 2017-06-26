#!/usr/bin/env python3

import os
import argparse
import sys

parser = argparse.ArgumentParser(description="This script get a vocab from unigram counts "
                                 "of words produced by get_unigram_counts.sh",
                                 epilog="E.g. " + sys.argv[0] + " data/rnnlm/data > data/rnnlm/vocab/words.txt",
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument("data_dir",
                    help="Directory in which to look for unigram counts.")

args = parser.parse_args()

eos_symbol = '</s>'
special_symbols = ['<s>', '<brk>', '<eps>']

def AddCounts(word_counts, text_file):
    with open(text_file, 'r', encoding="utf-8") as f:
        for line in f:
            line = line.strip()
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
        AddCounts(word_counts, full_path)

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
