#!/usr/bin/env python3

# Copyright  2017  Jian Wang
# License: Apache 2.0.

import os
import argparse
import sys

import re


parser = argparse.ArgumentParser(description="Validates features file, produced by rnnlm/choose_features.py.",
                                 epilog="E.g. " + sys.argv[0] + " exp/rnnlm/features.txt",
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument("features_file",
                    help="File containing features")

args = parser.parse_args()

EOS_SYMBOL = '</s>'

if not os.path.isfile(args.features_file):
    sys.exit(sys.argv[0] + ": Expected file {0} to exist".format(args.features_file))

with open(args.features_file, 'r', encoding="utf-8") as f:
    has_unigram = False
    has_length = False
    idx = 0
    match_feats = {}
    inital_feats = {}
    final_feats = {}
    word_feats = {}
    for line in f:
        fields = line.split()
        assert(len(fields) in [3, 4, 5])

        assert idx == int(fields[0])
        idx += 1

        # every feature should contain a scale
        scale = float(fields[-1])
        assert scale > 0.0 and scale <= 1.0

        if len(fields) == 3 and fields[1] == "length":
            if has_length:
                sys.exit(sys.argv[0] + ": Too many 'length' features")
            has_length = True
        else:
            if fields[1] == "constant":
                try:
                    assert len(fields) == 3
                    value = float(fields[2])
                    assert value > 0.0
                except:
                    sys.exit(sys.argv[0] + ": bad line: {0}".format(line))
            elif fields[1] == "special":
                if len(fields) != 4:
                    sys.exit(sys.argv[0] + ": bad line: {0}".format(line))
            elif fields[1] == "unigram":
                if float(fields[2]) <= 0.0:
                    sys.exit(sys.argv[0] + ": log-unigram-ppl should be a positive value: {0}".format(fields[2]))
                if has_unigram:
                    sys.exit(sys.argv[0] + ": Too many 'unigram' features")
                has_unigram = True
            elif fields[1] == "word":
                if fields[2] in word_feats:
                    sys.exit(sys.argv[0] + ": duplicated word feature: {0}".format(fields[2]))
                word_feats[fields[2]] = 1
            elif fields[1] == "initial":
                if fields[2] in inital_feats:
                    sys.exit(sys.argv[0] + ": duplicated initial feature: {0}".format(fields[2]))
                inital_feats[fields[2]] = 1
            elif fields[1] == "final":
                if fields[2] in final_feats:
                    sys.exit(sys.argv[0] + ": duplicated final feature: {0}".format(fields[2]))
                final_feats[fields[2]] = 1
            elif fields[1] == "match":
                if fields[2] in match_feats:
                    sys.exit(sys.argv[0] + ": duplicated match feature: {0}".format(fields[2]))
                match_feats[fields[2]] = 1
            else:
                sys.exit(sys.argv[0] + ": Error line format: {0}".format(line))
