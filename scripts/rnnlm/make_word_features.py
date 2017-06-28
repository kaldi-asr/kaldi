#!/usr/bin/env python3

import os
import argparse
import sys
import math


parser = argparse.ArgumentParser(description="This script turns the words into the sparse feature representation, "
                                             "using features from rnnlm/make_word_features.py.",
                                 epilog="E.g. " + sys.argv[0] + " --unigram-probs=exp/rnnlm/unigram_probs.txt "
                                        "data/rnnlm/vocab/words.txt exp/rnnlm/features.txt "
                                        "> exp/rnnlm/word_feats.txt",
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument("--unigram-probs", type=str, default='', required=True,
                    help="Specify the file containing unigram probs.")

parser.add_argument("vocab_file", help="Path for vocab file")
parser.add_argument("features_file", help="Path for features file")

args = parser.parse_args()


# read the voab
def read_vocab(vocab_file):
    vocab = {}
    with open(vocab_file, 'r', encoding="utf-8") as f:
        for line in f:
            fields = line.split()
            assert len(fields) == 2
            if fields[0] in vocab:
                sys.exit(sys.argv[0] + ": duplicated word({0}) in vocab: {1}"
                                       .format(fields[0], vocab_file))
            vocab[fields[0]] = int(fields[1])

    # check there is no duplication and no gap among word ids
    sorted_ids = sorted(vocab.values())
    assert len(sorted_ids) == len(vocab)
    for idx, id in enumerate(sorted_ids):
        assert idx == id

    return vocab


# read the unigram probs
def read_unigram_probs(unigram_probs_file):
    unigram_probs = []
    with open(unigram_probs_file, 'r', encoding="utf-8") as f:
        for line in f:
            fields = line.split()
            assert len(fields) == 2
            idx = int(fields[0])
            if idx >= len(unigram_probs):
                unigram_probs.extend([None] * (idx - len(unigram_probs) + 1))
            unigram_probs[idx] = float(fields[1])

    for prob in unigram_probs:
        assert prob is not None

    return unigram_probs


# read the features
# output is a dict with key can be 'special', 'unigram', 'length' and 'ngram',
#   feats['special'] is a dict whose key is special words and value is the feat_id
#   feats['unigram'] is a tuple with (feat_id, log_ppl)
#   feats['length']  is a int represents feat_id
#
#   feats['match']
#   feats['initial']
#   feats['final']
#   feats['word']    is a dict with key is ngram, value is feat_id for each type
#                    of ngram feature respectively.
#   feats['min_ngram_order'] is a int represents min-ngram-order
#   feats['max_ngram_order'] is a int represents max-ngram-order
def read_features(features_file):
    feats = {}
    feats['special'] = {}
    feats['match'] = {}
    feats['initial'] = {}
    feats['final'] = {}
    feats['word'] = {}
    feats['min_ngram_order'] = 10000
    feats['max_ngram_order'] = -1

    with open(features_file, 'r', encoding="utf-8") as f:
        for line in f:
            fields = line.split()
            assert len(fields) == 2 or len(fields) == 3

            feat_id = int(fields[0])
            feat_type = fields[1]
            if feat_type == 'special':
                feats['special'][fields[2]] = feat_id
            elif feat_type == 'unigram':
                feats['unigram'] = (feat_id, float(fields[2]))
            elif feat_type == 'length':
                feats['length'] = feat_id
            elif feat_type in ['word', 'match', 'initial', 'final']:
                ngram = fields[2]
                feats[feat_type][ngram] = feat_id
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
unigram_probs = read_unigram_probs(args.unigram_probs)
feats = read_features(args.features_file)

for word, idx in sorted(vocab.items(), key=lambda x: x[1]):
    print("{0}".format(idx), end="")

    if idx == 0:
        print("")
        continue

    prefix = "\t"
    if word in feats['special']:
        feat_id = feats['special'][word]
        print(prefix + "{0} 1".format(feat_id), end="")
        print("")
        continue

    if 'unigram' in feats:
        feat_id = feats['unigram'][0]
        log_ppl = feats['unigram'][1]
        logp = math.log(unigram_probs[idx])
        print(prefix + "{0} {1}".format(feat_id, logp / log_ppl + 1), end="")
        prefix = " "

    if 'length' in feats:
        feat_id = feats['length']
        print(prefix + "{0} {1}".format(feat_id, len(word)), end="")
        prefix = " "

    if word in feats['word']:
        feat_id = feats['word'][word]
        print(prefix + "{0} 1".format(feat_id), end="")
        prefix = " "

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
                feat_id = ngram_feats[feat]
                print(prefix + "{0} 1".format(feat_id), end="")
                prefix = " "

    print("")

print(sys.argv[0] + ": make features for {0} words.".format(len(vocab)), file=sys.stderr)
