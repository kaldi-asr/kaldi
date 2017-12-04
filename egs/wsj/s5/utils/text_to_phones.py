#!/usr/bin/env python

# Copyright 2017 Johns Hopkins University (author: Hossein Hadian)
# Apache 2.0


""" This reads data/train/text from std input and converts the word transcriptions
    to phone transcriptions
"""

import argparse
import os
import sys
import copy
import math
import random

parser = argparse.ArgumentParser(description="""This script ...""")
parser.add_argument('langdir', type=str)
parser.add_argument('lex', type=str,
                    help='lexicon.txt')
parser.add_argument('--be-silprob', type=float, default=0.8,
                    help="""Probability of optional silence at the beginning
                    and end.""")
parser.add_argument('--silprob', type=float, default=0.2,
                    help="Probability of optional silence between the words.")
parser.add_argument('--pos-indep', action='store_true')


args = parser.parse_args()

# optional silence
sil = open(args.langdir + "/phones/optional_silence.txt").readline().strip()
#print(sil)

# unk
unk = open(args.langdir + "/oov.txt").readline().strip()
#print(unk)

# load the lexicon
lex = {}
with open(args.lex) as f:
    for line in f:
        line = line.strip();
        parts = line.split()
        lex[parts[0]] = parts[1:]

#i = 0
#for w in lex:
#    if i < 10:
#        print("{} --> {}".format(w, lex[w]))
#    i +=1

n_tot = 0
n_fail = 0
for line in sys.stdin:
    line = line.strip().split()
    key = line[0]
    wtrans = line[1:]
    ptrans = []
    if random.random() < args.be_silprob:
        ptrans += [sil]
    for i in range(len(wtrans)):
        n_tot += 1
        w = wtrans[i]
        if w not in lex:
            n_fail += 1
            if n_fail < 20:
                sys.stderr.write("{} not found in lexicon, replacing with {}\n".format(w, unk))
            elif n_fail == 20:
                sys.stderr.write("Not warning about OOVs any more.\n")
            tr = lex[unk]
        else:
            #print("{}   -->   {}".format(w, lex[w]))
            tr = copy.deepcopy(lex[w])
            if not args.pos_indep:
                if len(tr) == 1:
                    tr[0] += "_S"
                else:
                    tr[0] += "_B"
                    tr[-1] += "_E"
                    for j in range(1, len(tr) - 1):
                        tr[j] += "_I"
        ptrans += tr
        p = args.silprob if i < len(wtrans) - 1 else args.be_silprob
        if random.random() < p:
            ptrans += [sil]
    print(key + " " + " ".join(ptrans))

sys.stderr.write("{} out of {} were OOVs\n".format(n_fail, n_tot))
