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
def ReadVocab(vocab_file):
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
def ReadUnigramProbs(unigram_probs_file):
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
        assert not prob is None

    return unigram_probs

# read the features
# output is a dict with key can be 'special', 'unigram', 'length' and 'ngram',
#   feats['special'] is a dict whose key is special words and value is the feat_id
#   feats['unigram'] is a tuple with (feat_id, log_ppl)
#   feats['length']  is a int represents feat_id
#   feats['ngram']   is a list of tuple with (feat_id, feat_type, ngram),
#                    where feat_type can be 'match', 'initial', 'final' or 'word',
#                    ngram is the substr used to compare
def ReadFeatures(features_file):
    feats = {}
    feats['special'] = {}
    feats['ngram'] = []

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
                feats['ngram'].append((feat_id, feat_type, fields[2]))
            else:
                sys.exit(sys.argv[0] + ": error feature type: {0}".format(feat_type))

    return feats

vocab = ReadVocab(args.vocab_file)
unigram_probs = ReadUnigramProbs(args.unigram_probs)
feats = ReadFeatures(args.features_file)

for word, idx in sorted(vocab.items(), key=lambda x:x[1]):
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

    for ngram_feat in feats['ngram']:
        feat_id = ngram_feat[0]
        feat_type = ngram_feat[1]
        ngram = ngram_feat[2]

        active = False
        if feat_type == "match":
            if ngram in word:
                active = True
        elif feat_type == "initial":
            if len(word) >= len(ngram) and word[:len(ngram)] == ngram:
                active = True
        elif feat_type == "final":
            if len(word) >= len(ngram) and word[len(word)-len(ngram):] == ngram:
                active = True
        elif feat_type == "word":
            if word == ngram:
                active = True
        else:
            sys.exit(sys.argv[0] + ": error feature type: {0}".format(feat_type))

        if active:
            print(prefix + "{0} 1".format(feat_id), end="")
            prefix = " "

    print("")

print(sys.argv[0] + ": make features for {0} words.".format(len(vocab)), file=sys.stderr)
