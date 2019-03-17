#!/usr/bin/env python3

# Copyright  2017  Jian Wang
# License: Apache 2.0.

import os
import argparse
import sys

import re
tab_or_space = re.compile('[ \t]+')

parser = argparse.ArgumentParser(description="Validates word features file, produced by rnnlm/get_word_features.py.",
                                 epilog="E.g. " + sys.argv[0] + " --features-file=exp/rnnlm/features.txt "
                                        "exp/rnnlm/word_feats.txt",
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument("--features-file", type=str, default='', required=True,
                    help="File containing features")
parser.add_argument("word_features_file", help="File containing word features")

args = parser.parse_args()

# we only need to know the feat_id for 'special', 'unigram' and 'length'
special_feat_ids = []
constant_feat_id = -1
constant_feat_value = None
unigram_feat_id = -1
length_feat_id = -1
max_feat_id = -1
with open(args.features_file, 'r', encoding="latin-1") as f:
    for line in f:
        fields = re.split(tab_or_space, line)
        assert(len(fields) in [3, 4, 5])

        feat_id = int(fields[0])

        # every feature should contain a scale
        scale = float(fields[-1])
        assert scale > 0.0 and scale <= 1.0

        if fields[1] == "special":
            special_feat_ids.append(feat_id)
        elif fields[1] == "constant":
            constant_feat_id = feat_id
            constant_feat_value = scale
        elif fields[1] == "unigram":
            unigram_feat_id = feat_id
        elif fields[1] == "length":
            length_feat_id = feat_id

        if feat_id > max_feat_id:
            max_feat_id = feat_id

with open(args.word_features_file, 'r', encoding="latin-1") as f:
    for line in f:
        fields = re.split(tab_or_space, line)
        assert len(fields) > 0 and len(fields) % 2 == 1
        word_id = int(fields[0])

        if len(fields) == 1:
            if word_id != 0:
                sys.exit(sys.argv[0] + ": Only <eps> can have no feature: {0}.".format(line))
        i = 1
        last_feat_id = -1
        while i < len(fields):
            feat_id = int(fields[i])
            feat_value = fields[i + 1]
            if feat_id <= last_feat_id:
                sys.exit(sys.argv[0] + ": features must be listed in increasing order: {0} <= {1} in {2}.".format(feat_id, last_feat_id, line))
            last_feat_id = feat_id

            if feat_id > max_feat_id:
                sys.exit(sys.argv[0] + ": Wrong feat_id: {0}.".format(line))
            elif feat_id in special_feat_ids:
                if len(fields) != 3 and len(fields) != 5:
                    sys.exit(sys.argv[0] + ": Special word can only have one or 2 features: {0}.".format(line))
                try:
                    float(feat_value)
                except ValueError:
                    sys.exit(sys.argv[0] + ": Value of special word feature should be a float number: {0}.".format(line))
            elif feat_id == constant_feat_id:
                if abs(float(feat_value) - constant_feat_value) > 1e-6:
                    sys.exit(sys.argv[0] + ": Value of constant feature is not right: {0}".format(line))
            else: # all feat_value would be float
                try:
                    float(feat_value)
                except ValueError:
                    sys.exit(sys.argv[0] + ": Value of unigram feature should be a float number: {0}.".format(line))
            i += 2
