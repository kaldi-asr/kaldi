#!/usr/bin/env python3

# Copyright  2017  Jian Wang
# License: Apache 2.0.

import os
import argparse
import sys

# The use of latin-1 encoding does not preclude reading utf-8.  latin-1 encoding
# means "treat words as sequences of bytes", and it is compatible with utf-8
# encoding as well as other encodings such as gbk, as long as the spaces are
# also spaces in ascii (which we check).  It is basically how we emulate the
# behavior of python before python3.
sys.stdout = open(1, 'w', encoding='latin-1', closefd=False)

import re
tab_or_space = re.compile('[ \t]+')

parser = argparse.ArgumentParser(description="This script turns the word features to a human readable format.",
                                 epilog="E.g. " + sys.argv[0] + "exp/rnnlm/word_feats.txt exp/rnnlm/features.txt "
                                        "> exp/rnnlm/word_feats.str.txt",
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument("word_features_file", help="Path for word_feats file")
parser.add_argument("features_file", help="Path for features file")

args = parser.parse_args()


# read the feature types
# return a dict mapping the feat_id to a tuple of
# (feat_type, feat_key)
# feat_key is the string for 'special', 'word', 'initial', 'match', 'final'
# features
def read_feature_type_and_key(features_file):
    feat_types = {}

    with open(features_file, 'r', encoding="latin-1") as f:
        for line in f:
            fields = re.split(tab_or_space, line)
            assert(len(fields) in [2, 3, 4])

            feat_id = int(fields[0])
            feat_type = fields[1]
            feat_key = ''
            if feat_type in ['special', 'word', 'initial', 'match', 'final']:
                feat_key = fields[2]
            feat_types[feat_id] = (feat_type, feat_key)

    return feat_types

feat_type_and_key = read_feature_type_and_key(args.features_file)

num_word_feats = 0
with open(args.word_features_file, 'r', encoding="latin-1") as f:
    for line in f:
        fields = re.split(tab_or_space, line)
        assert len(fields) % 2 == 1

        print(int(fields[0]), end='\t')
        for idx in range(1, len(fields), 2):
          feat_id = int(fields[idx])
          feat_value = fields[idx + 1]
          feat_type, feat_key = feat_type_and_key[feat_id]
          if feat_type == 'constant':
              print(' "constant"={0}'.format(feat_value), end='')
          elif feat_type == 'unigram':
              print(' "unigram"={0}'.format(feat_value), end='')
          elif feat_type == 'length':
              print(' "length"={0}'.format(feat_value), end='')
          else: # other types are the same
              print(' "{0} {1}"={2}'.format(feat_type, feat_key, feat_value), end='')
        print('')
        num_word_feats += 1


print(sys.argv[0] + ": show features for {0} words.".format(num_word_feats), file=sys.stderr)
