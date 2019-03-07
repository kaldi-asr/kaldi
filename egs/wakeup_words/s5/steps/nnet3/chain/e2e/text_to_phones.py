#!/usr/bin/env python

# Copyright    2017 Hossein Hadian
# Apache 2.0


""" This reads data/train/text from standard input, converts the word transcriptions
    to phone transcriptions using the provided lexicon,
    and writes them to standard output.
"""
from __future__ import print_function

import argparse
from os.path import join
import sys
import copy
import random

parser = argparse.ArgumentParser(description="""This script reads
    data/train/text from std input and converts the word transcriptions
    to phone transcriptions using the provided lexicon""")
parser.add_argument('langdir', type=str)
parser.add_argument('--edge-silprob', type=float, default=0.8,
                    help="""Probability of optional silence at the beginning
                    and end.""")
parser.add_argument('--between-silprob', type=float, default=0.2,
                    help="Probability of optional silence between the words.")


args = parser.parse_args()

# optional silence
sil = open(join(args.langdir,
                "phones/optional_silence.txt")).readline().strip()

oov_word = open(join(args.langdir, "oov.txt")).readline().strip()


# load the lexicon
lexicon = {}
with open(join(args.langdir, "phones/align_lexicon.txt")) as f:
    for line in f:
        line = line.strip();
        parts = line.split()
        lexicon[parts[0]] = parts[2:]  # ignore parts[1]

n_tot = 0
n_fail = 0
for line in sys.stdin:
    line = line.strip().split()
    key = line[0]
    word_trans = line[1:]   # word-level transcription
    phone_trans = []        # phone-level transcription
    if random.random() < args.edge_silprob:
        phone_trans += [sil]
    for i in range(len(word_trans)):
        n_tot += 1
        word = word_trans[i]
        if word not in lexicon:
            n_fail += 1
            if n_fail < 20:
                sys.stderr.write("{} not found in lexicon, replacing with {}\n".format(word, oov_word))
            elif n_fail == 20:
                sys.stderr.write("Not warning about OOVs any more.\n")
            pronunciation = lexicon[oov_word]
        else:
            pronunciation = copy.deepcopy(lexicon[word])
        phone_trans += pronunciation
        prob = args.between_silprob if i < len(word_trans) - 1 else args.edge_silprob
        if random.random() < prob:
            phone_trans += [sil]
    print(key + " " + " ".join(phone_trans))

sys.stderr.write("Done. {} out of {} were OOVs.\n".format(n_fail, n_tot))
