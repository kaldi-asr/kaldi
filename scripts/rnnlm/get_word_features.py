#!/usr/bin/env python3

# Copyright  2017  Jian Wang
# License: Apache 2.0.

import os
import argparse
import sys
import math
from collections import defaultdict

import re
tab_or_space = re.compile('[ \t]+')

parser = argparse.ArgumentParser(description="This script turns the words into the sparse feature representation, "
                                             "using features from rnnlm/choose_features.py.",
                                 epilog="E.g. " + sys.argv[0] + " --unigram-probs=exp/rnnlm/unigram_probs.txt "
                                        "data/rnnlm/vocab/words.txt exp/rnnlm/features.txt "
                                        "> exp/rnnlm/word_feats.txt",
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument("--unigram-probs", type=str, default='',
                    help="Specify the file containing unigram probs.")
parser.add_argument("vocab_file", help="Path for vocab file")
parser.add_argument("features_file", help="Path for features file")
parser.add_argument("--treat-as-bos", type=str, default='',
                    help="""Comma-separated list of written representations of
                    words that are to be treated the same as the BOS symbol
                    <s> for purposes of getting the word features (i.e. they will
                    have the same features as <s>.  Because <s> will always
                    learn to be predicted with a close-to-zero probability, this is
                    a suitable thing to do for words that are in words.txt but
                    are never expected to be predicted.  (Note: it's not necessary
                    to do this for symbol zero, <eps>, because we exclude it from
                    the normalization sum).  Example: --treat-as-bos='#0'""")

args = parser.parse_args()


# read the voab
# return the vocab, which is a dict mapping the word to a integer id.
def read_vocab(vocab_file):
    vocab = {}
    with open(vocab_file, 'r', encoding="latin-1") as f:
        for line in f:
            fields = re.split(tab_or_space, line)
            assert len(fields) == 2
            if fields[0] in vocab:
                sys.exit(sys.argv[0] + ": duplicated word({0}) in vocab: {1}"
                                       .format(fields[0], vocab_file))
            vocab[fields[0]] = int(fields[1])

    # check there is no duplication and no gap among word ids
    sorted_ids = sorted(vocab.values())
    for idx, id in enumerate(sorted_ids):
        assert idx == id

    return vocab


# read the unigram probs
# return a list of unigram_probs, indexed by word id
def read_unigram_probs(unigram_probs_file):
    unigram_probs = []
    with open(unigram_probs_file, 'r', encoding="latin-1") as f:
        for line in f:
            fields = re.split(tab_or_space, line)
            assert len(fields) == 2
            idx = int(fields[0])
            if idx >= len(unigram_probs):
                unigram_probs.extend([None] * (idx - len(unigram_probs) + 1))
            unigram_probs[idx] = float(fields[1])

    for prob in unigram_probs:
        assert prob is not None

    return unigram_probs


# read the features
# return a dict with following items:

#   feats['constant'] is None if there is no constant feature used, else
#                     a 2-tuple (feat_id, value), e.g. (1, 0.01)
#   feats['special'] is a dict whose key is special words and value is a tuple (feat_id, scale)
#   feats['unigram'] is a tuple with (feat_id, entropy, scale)
#   feats['length']  is a tuple with (feat_id, scale)
#
#   feats['match']
#   feats['initial']
#   feats['final']
#   feats['word']    is a dict with key is ngram, value is a tuple (feat_id, scale)
#   feats['min_ngram_order'] is a int represents min-ngram-order
#   feats['max_ngram_order'] is a int represents max-ngram-order
def read_features(features_file):
    feats = {}
    feats['constant'] = None
    feats['special'] = {}
    feats['match'] = {}
    feats['initial'] = {}
    feats['final'] = {}
    feats['word'] = {}
    feats['min_ngram_order'] = 10000
    feats['max_ngram_order'] = -1

    with open(features_file, 'r', encoding="latin-1") as f:
        for line in f:
            fields = re.split(tab_or_space, line)
            assert(len(fields) in [3, 4, 5])

            feat_id = int(fields[0])
            feat_type = fields[1]
            scale = float(fields[-1])
            if feat_type == 'constant':
                value = float(fields[2])
                feats['constant'] = (feat_id, value)
            elif feat_type == 'special':
                feats['special'][fields[2]] = (feat_id, scale)
            elif feat_type == 'unigram':
                feats['unigram'] = (feat_id, float(fields[2]), scale)
            elif feat_type == 'length':
                feats['length'] = (feat_id, scale)
            elif feat_type in ['word', 'match', 'initial', 'final']:
                ngram = fields[2]
                feats[feat_type][ngram] = (feat_id, scale)
                if feat_type == 'word':
                    continue
                elif feat_type in ['initial', 'final']:
                    order = len(ngram) + 1
                else:
                    order = len(ngram)
                if order > feats['max_ngram_order']:
                    feats['max_ngram_order'] = order
                if order < feats['min_ngram_order']:
                    feats['min_ngram_order'] = order
            else:
                sys.exit(sys.argv[0] + ": error feature type: {0}".format(feat_type))

    return feats

vocab = read_vocab(args.vocab_file)
if args.unigram_probs != '':
    unigram_probs = read_unigram_probs(args.unigram_probs)
else:
    unigram_probs = None
feats = read_features(args.features_file)

treat_as_bos_word_set = args.treat_as_bos.split(',')

def treat_as_bos(word):
  return word in treat_as_bos_word_set

def get_feature_list(word, idx):
    """Return a dict from feat_id to value (as int or float), e.g.
      { 0 -> 1.0, 100 -> 1 }
    """
    ans = defaultdict(int)  # the default is only used for character-ngram features.
    if idx == 0:
        return ans

    if feats['constant'] is not None:
        (feat_id, value) = feats['constant']
        ans[feat_id] = value

    if word in feats['special']:
        (feat_id, scale) = feats['special'][word]
        ans[feat_id] = 1 * scale
        return ans   # return because words with the 'special' feature do
                     # not get any other features (except the constant
                     # feature).

    if 'unigram' in feats:
        if unigram_probs is None:
            sys.exit(sys.argv[0] + ": if unigram feature is present, you must specify the "
                     "--unigram-probs option.");
        (feat_id, offset, scale) = feats['unigram']
        logp = math.log(unigram_probs[idx])
        ans[feat_id] = offset + logp * scale

    if 'length' in feats:
        (feat_id, scale) = feats['length']
        ans[feat_id] = len(word) * scale

    if word in feats['word']:
        (feat_id, scale) = feats['word'][word]
        ans[feat_id] = 1 * scale

    for pos in range(len(word) + 1):  # +1 for EOW
        for order in range(feats['min_ngram_order'], feats['max_ngram_order'] + 1):
            start = pos - order + 1
            end = pos + 1

            if start < -1:
                continue

            if start < 0 and end > len(word):
                # 'word' feature, which we already match before
                continue
            elif start < 0:
                ngram_feats = feats['initial']
                start = 0
            elif end > len(word):
                ngram_feats = feats['final']
                end = len(word)
            else:
                ngram_feats = feats['match']
            if start >= end:
                continue

            feat = word[start:end]
            if feat in ngram_feats:
                (feat_id, scale) = ngram_feats[feat]
                ans[feat_id] += 1 * scale
    return ans

for word, idx in sorted(vocab.items(), key=lambda x: x[1]):
    if treat_as_bos(word):
      feature_list = get_feature_list("<s>", idx)
    else:
      feature_list = get_feature_list(word, idx)
    print("{0}\t{1}".format(idx,
                            " ".join(["%s %.3g" % (f, v) for f, v in sorted(feature_list.items())])))

print(sys.argv[0] + ": made features for {0} words.".format(len(vocab)), file=sys.stderr)
