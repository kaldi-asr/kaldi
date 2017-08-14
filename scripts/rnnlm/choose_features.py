#!/usr/bin/env python3

# Copyright  2017  Jian Wang
# License: Apache 2.0.

import os
import argparse
import sys
import math
sys.stdout = open(1, 'w', encoding='utf-8', closefd=False)


parser = argparse.ArgumentParser(description="This script chooses the sparse feature representation of words. "
                                             "To be more specific, it chooses the set of features-- you compute "
                                             "them for the specific words by calling rnnlm/make_word_features.py.",
                                 epilog="E.g. " + sys.argv[0] + " --unigram-probs=exp/rnnlm/unigram_probs.txt "
                                        "--unigram-scale=0.1 "
                                        "data/rnnlm/vocab/words.txt > exp/rnnlm/features.txt",
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument("--unigram-probs", type=str, default='', required=True,
                    help="Specify the file containing unigram probs.")
parser.add_argument("--min-ngram-order", type=int, default=2,
                    help="minimum length of n-grams of characters to"
                         "make potential features.")
parser.add_argument("--max-ngram-order", type=int, default=3,
                    help="maximum length of n-grams of characters to"
                         "make potential features.")
parser.add_argument("--min-frequency", type=float, default=1.0e-05,
                    help="minimum frequency with which an n-gram character "
                         "feature is encountered (counted as binary presence in a word times unigram "
                         "probs of words), for it to be used as a feature. e.g. "
                         "if less than 1.0e-06 of tokens contain the n-gram 'xyz', "
                         "then it wouldn't be used as a feature.")
parser.add_argument("--include-unigram-feature", type=str, default='true',
                    choices=['true', 'false'],
                    help="If true, the unigram frequency of a word is "
                         "one of the features.  [note: one reason we "
                         "want to include this, is to make it easier to "
                         "port models to new vocabularies and domains].")
parser.add_argument("--include-length-feature", type=str, default='true',
                    choices=['true', 'false'],
                    help="If true, the length in characters of a word is one of the features.")
parser.add_argument("--top-word-features", type=int, default=2000,
                    help="The most frequent n words each get their own "
                         "special feature, in addition to any other features "
                         "that the word may naturally get.")
parser.add_argument("--special-words", type=str, default='<s>,</s>,<brk>',
                    help="List of special words that get their own special "
                         "features and do not get any other features.")
parser.add_argument("--constant-feature", type=float,
                    help="If supplied, the value of a feature that all words "
                    "are given.  Helps model offsets.  E.g. 1.0.")
parser.add_argument("--max-feature-rms", type=float, default=0.01,
                    help="maximum allowed root-mean-square value for any feature.")

parser.add_argument("vocab_file",
                    help="Path for vocab file")

args = parser.parse_args()

if args.min_ngram_order < 1:
    sys.exit(sys.argv[0] + ": --min-ngram-order must be at least 1.")
if args.max_ngram_order < args.min_ngram_order:
    sys.exit(sys.argv[0] + ": --max-ngram-order must be larger than or equal to --min-ngram-order.")

SPECIAL_SYMBOLS = ["<eps>", "<s>", "<brk>"]


# read the voab
# return the vocab, which is a dict mapping the word to a integer id.
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

def get_feature_scale(rms):
    if rms > args.max_feature_rms:
        return args.max_feature_rms / rms
    else:
        return 1.0

vocab = read_vocab(args.vocab_file)
wordlist = [x[0] for x in sorted(vocab.items(), key=lambda x:x[1])]
unigram_probs = read_unigram_probs(args.unigram_probs)
assert len(unigram_probs) == len(wordlist)

vocab_size = len(vocab)
if '<eps>' in vocab:
    vocab_size -= 1

num_features = 0

if args.constant_feature is not None:
    print("{0}\tconstant\t{1}\t{2}".format(num_features, args.constant_feature,
          get_feature_scale(args.constant_feature)))
    num_features += 1

# special words features
if args.special_words != '':
    for word in args.special_words.split(','):
        rms = math.sqrt(1.0 / vocab_size)
        print("{0}\tspecial\t{1}\t{2}".format(num_features, word, get_feature_scale(rms)))
        num_features += 1

# unigram features
if args.include_unigram_feature == 'true':
    entropy = 0.0
    entropy_total = 0.0
    for idx, p in enumerate(unigram_probs):
        if wordlist[idx] in SPECIAL_SYMBOLS:
            continue
        if p > 0.0:
            entropy += math.log(p) * p
            entropy_total += p
    entropy /= -entropy_total

    rms = 0.0
    for idx, p in enumerate(unigram_probs):
        if wordlist[idx] in SPECIAL_SYMBOLS:
            continue
        if p > 0.0:
            rms += (p - entropy) ** 2
    rms = math.sqrt(rms / vocab_size)

    print("{0}\tunigram\t{1}\t{2}".format(num_features, entropy, get_feature_scale(rms)))
    num_features += 1

# length features
if args.include_length_feature == 'true':
    rms = 0.0
    for word in wordlist:
        if word in SPECIAL_SYMBOLS + ["</s>"]:
            continue
        rms += len(word) ** 2
    rms = math.sqrt(rms / vocab_size)
    print("{0}\tlength\t{1}".format(num_features, get_feature_scale(rms)))
    num_features += 1

# top words features
top_words = {}
if args.top_word_features > 0:
    sorted_words = sorted(zip(wordlist, unigram_probs), key=lambda x: x[1], reverse=True)
    num_words = 0
    for word, _ in sorted_words:
        if word in SPECIAL_SYMBOLS + ["</s>"]:
            continue

        rms = math.sqrt(1.0 / vocab_size)
        print("{0}\tword\t{1}\t{2}".format(num_features, word, get_feature_scale(rms)))
        num_features += 1

        top_words[word] = 1
        num_words += 1
        if num_words > args.top_word_features:
            break

# n-gram features
# these dict mapping a ngram to a tuple (word_freq_sum, word_count_sum)
initial_feats = {}
final_feats = {}
match_feats = {}
word_feats = {}
for word in wordlist:
    if word in SPECIAL_SYMBOLS + ["</s>"]:
        continue

    word_freq = unigram_probs[vocab[word]]
    for pos in range(len(word) + 1):  # +1 for EOW
        for order in range(args.min_ngram_order, args.max_ngram_order + 1):
            start = pos - order + 1
            end = pos + 1

            if start < -1:
                continue

            if start < 0 and end > len(word):
                feats = word_feats
                start = 0
                end = len(word)
            elif start < 0:
                feats = initial_feats
                start = 0
            elif end > len(word):
                feats = final_feats
                end = len(word)
            else:
                feats = match_feats
            if start >= end:
                continue

            feat = word[start:end]
            if feat in feats:
                feats[feat] = (feats[feat][0] + word_freq,
                               feats[feat][1] + 1)
            else:
                feats[feat] = (word_freq, 1)

for feat, (freq, square_freq) in initial_feats.items():
    if freq < args.min_frequency:
        continue
    rms = math.sqrt(1.0 * square_freq / vocab_size)
    print("{0}\tinitial\t{1}\t{2}".format(num_features, feat, get_feature_scale(rms)))
    num_features += 1

for feat, (freq, count) in final_feats.items():
    if freq < args.min_frequency:
        continue
    rms = math.sqrt(1.0 * count / vocab_size)
    print("{0}\tfinal\t{1}\t{2}".format(num_features, feat, get_feature_scale(rms)))
    num_features += 1

for feat, (freq, count) in match_feats.items():
    if freq < args.min_frequency:
        continue
    rms = math.sqrt(1.0 * count / vocab_size)
    print("{0}\tmatch\t{1}\t{2}".format(num_features, feat, get_feature_scale(rms)))
    num_features += 1

for feat, (freq, count) in word_feats.items():
    if freq < args.min_frequency or feat in top_words:
        continue
    rms = math.sqrt(1.0 * count / vocab_size)
    print("{0}\tword\t{1}\t{2}".format(num_features, feat, get_feature_scale(rms)))
    num_features += 1

print(sys.argv[0] + ": chose {0} features.".format(num_features), file=sys.stderr)
