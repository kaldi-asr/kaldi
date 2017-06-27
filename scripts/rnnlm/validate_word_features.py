#!/usr/bin/env python3

import os
import argparse
import sys

parser = argparse.ArgumentParser(description="Validates word features file, produced by rnnlm/make_word_features.py.",
                                 epilog="E.g. " + sys.argv[0] + " --features-file=exp/rnnlm/features.txt "
                                        "exp/rnnlm/word_feats.txt",
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument("--features-file", type=str, default='', required=True,
                    help="File containing features")
parser.add_argument("word_features_file", help="File containing word features")

args = parser.parse_args()

# we only need to know the feat_id for 'special', 'unigram' and 'length'
special_feat_ids = []
unigram_feat_id = -1
length_feat_id = -1
max_feat_id = -1
with open(args.features_file, 'r', encoding="utf-8") as f:
    for line in f:
        fields = line.split()
        assert len(fields) == 2 or len(fields) == 3

        feat_id = int(fields[0])
        if fields[1] == "special":
            special_feat_ids.append(feat_id)
        elif fields[1] == "unigram":
            unigram_feat_id = feat_id
        elif fields[1] == "length":
            length_feat_id = feat_id

        if feat_id > max_feat_id:
            max_feat_id = feat_id

with open(args.word_features_file, 'r', encoding="utf-8") as f:
    for line in f:
        fields = line.split()
        assert len(fields) > 0 and len(fields) % 2 == 1
        word_id = int(fields[0])

        if len(fields) == 1:
            if word_id != 0:
                sys.exit(sys.argv[0] + ": Only <eps> can have no feature: {0}.".format(line))
        i = 1
        while i < len(fields):
            feat_id = int(fields[i])
            feat_value = fields[i + 1]
            if feat_id in special_feat_ids:
                if len(fields) != 3:
                    sys.exit(sys.argv[0] + ": Special word can only have one feature: {0}.".format(line))
                if int(feat_value) != 1:
                    sys.exit(sys.argv[0] + ": Value of special word feature must be 1: {0}.".format(line))
            elif feat_id == unigram_feat_id:
                try:
                    float(feat_value)
                except ValueError:
                    sys.exit(sys.argv[0] + ": Value of unigram feature should be a float number: {0}.".format(line))
            elif feat_id == length_feat_id:
                try:
                    int(feat_value)
                except ValueError:
                    sys.exit(sys.argv[0] + ": Value of length feature should be a integer number: {0}.".format(line))
            else:
                if feat_id > max_feat_id:
                    sys.exit(sys.argv[0] + ": Wrong feat_id: {0}.".format(line))
                if int(feat_value) != 1:
                    sys.exit(sys.argv[0] + ": Value of ngram feature must be 1: {0}.".format(line))
            i += 2
