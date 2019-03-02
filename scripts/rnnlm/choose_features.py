#!/usr/bin/env python3

# Copyright  2017  Jian Wang
# License: Apache 2.0.

import os
import argparse
import sys
import math
from collections import defaultdict
sys.stdout = open(1, 'w', encoding='utf-8', closefd=False)

# because this script splits inside words, we cannot use latin-1; we actually need to know what 
# what the encoding is.  By default we make this utf-8; to handle encodings that are not compatible
# with utf-8 (e.g. gbk), we'll eventually have to make the encoding an option to this script.

import re
tab_or_space = re.compile('[ \t]+')

parser = argparse.ArgumentParser(description="This script chooses the sparse feature representation of words. "
                                             "To be more specific, it chooses the set of features-- you compute "
                                             "them for the specific words by calling rnnlm/get_word_features.py.",
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
parser.add_argument("--use-constant-feature", type=str, default="false",
                    help="If set to true, we give a constant feature to all "
                    "words (to help model offsets).  The value will equal "
                    "the --max-feature-rms option.");
parser.add_argument("--max-feature-rms", type=float, default=0.01,
                    help="maximum allowed root-mean-square value for any feature.")

# dir=exp/rnnlm_tdnn_d
# paste <(awk '{print $2}' $dir/config/unigram_probs.txt) <(awk '{$1="";print;}' $dir/word_feats.txt ) | awk '{freq=$1; num_feats=(NF-1)/2; for (n=1;n<=num_feats;n++) { a=n*2; b=n*2+1; rms[$a] += freq * $b*$b; }} END{for(k in rms) { print k, rms[k];}}' | sort -k2 -nr | head
# 7180 9.99985e-05
# 7019 9.99947e-05
# ..

parser.add_argument("vocab_file",
                    help="Path for vocab file")

args = parser.parse_args()

if args.use_constant_feature != "false" and args.use_constant_feature != "true":
    sys.exit(sys.argv[0] + ": --use-constant-feature must be true or false: {0}".format(
        args.use_constant_feature))
if args.min_ngram_order < 1:
    sys.exit(sys.argv[0] + ": --min-ngram-order must be at least 1.")
if args.max_ngram_order < args.min_ngram_order:
    sys.exit(sys.argv[0] + ": --max-ngram-order must be larger than or equal to --min-ngram-order.")

SPECIAL_SYMBOLS = ["<eps>", "<s>", "<brk>"]


# read the vocabulay file
# Returns a pair (vocab, wordlist)
# where 'vocab' is a dict mapping the string-valued word to a integer id.
#  and 'wordlist' is a list indexed by integer id, that returns the string-valued word.
def read_vocab(vocab_file):
    vocab = {}
    with open(vocab_file, 'r', encoding="utf-8") as f:
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

    vocab_size = 1 + max(vocab.values())
    wordlist = [ None] * vocab_size
    for word, index in vocab.items():
        assert wordlist[index] is None
        wordlist[index] = word

    if wordlist[0] != '<eps>' and wordlist[0] != '<EPS>':
        sys.exit(sys.argv[0] + ": expected word numbered zero to be epsilon.")
    return (vocab, wordlist)


# read the unigram probs; returns a list indexed by integer
# id of the word, which evaluates to the unigram prob of the word.
def read_unigram_probs(unigram_probs_file):
    unigram_probs = []
    with open(unigram_probs_file, 'r', encoding="utf-8") as f:
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

def get_feature_scale(rms):
    if rms > args.max_feature_rms:
        return '%.2g' % (args.max_feature_rms / rms)
    else:
        return "1.0"

(vocab, wordlist) = read_vocab(args.vocab_file)
unigram_probs = read_unigram_probs(args.unigram_probs)
assert len(unigram_probs) == len(wordlist)

# num_features is a counter used to keep track of how many features
# we've created so far.
num_features = 0

# The constant feature; this will be a line of the form
# <feature-index> constant feature-value
# e.g.
# constant 0.01
# which means every single word gets this feature.  It is used to
# handle offsets of the words' log-likelihoods, so we don't have
# to include an offset term in the math.
if args.use_constant_feature == "true":
    print("{0}\tconstant\t{1}".format(num_features,
                                      args.max_feature_rms))
    num_features += 1


# 'word_indexes_to_exclude' will contain the integer indexes of words that are
# in 'args.special_words' plus the zero word (epsilon) and which don't take part
# in later-numbered features.
word_indexes_to_exclude = {0} # a set including only zero.

# Features for 'special' words, i.e. a line of the form
# <feature-index> special <word> <feature-value>
# e.g.:
# 2 special </s> 1.0
# These words get just the constant feature (if present) and their own 'special'
# feature, but not letter-based features, because things like '<s>' are
# special symbols that are not treated as regular words.
if args.special_words != '':
    for word in args.special_words.split(','):
        if not word in vocab:
            sys.exit(sys.argv[0] + ": error: element {0} of --special-words option "
                     "is not in the vocabulary file {1}".format(word, args.vocab_file))
        word_indexes_to_exclude.add(vocab[word])
        this_word_prob = unigram_probs[vocab[word]]
        rms = math.sqrt(this_word_prob)
        print("{0}\tspecial\t{1}\t{2}".format(num_features, word,
                                              get_feature_scale(rms)))
        num_features += 1


# Print a line for the unigram feature (this is a feature that's a scaled,
# offset version of the log-unigram-prob of the word).  The line is of the form:
# <feature-index> unigram <offset> <scale>
# e.g.
# 3 unigram 0.04631 0.024312
# where the interpretation is that if a word w has unigram probability p(w), the
# feature's value will equal <offset> + <scale> * log(p(w)).

# The offset and scale are chosen so that the expected value of the feature
# is zero and its rms value equals args.max_feature_rms.
if args.include_unigram_feature == 'true':
    total_p = 0.0  # total probability of words that have the unigram feature,
                   # i.e. excluding words with the 'special' feature.
    total_x = 0.0
    total_x2 = 0.0
    for idx, p in enumerate(unigram_probs):
        if p > 0.0 and idx not in word_indexes_to_exclude:
            # 'feature_value' is the value of the log-unigram-prob feature
            # before the offset and scale are accounted for.  We accumulate
            # the expected x and x^2 stats of this.
            feature_value = math.log(p)
            total_p += p
            total_x += p * feature_value
            total_x2 += p * feature_value * feature_value
    # we won't allow all the words to be 'special' words.
    # total_p is the probability mass of non-special words.
    assert total_p > 0 and total_p < 1.01
    mean = total_x / total_p
    variance = (total_x2 / total_p - mean * mean)
    # The following assert is because training an RNNLM with only one
    # 'non-special' word and the unigram feature (or using the unigram feature
    # where all unigram probs are the same) does not make sense.
    assert variance > 0
    # The mean is computed over those words where the feature was present..
    # 'stddev' is computed over all words, even when the feature was not present
    # (that's what the factor of 'total_p' is about); this is consistent with
    # how we apply the args.max_feature_rms option in general.
    stddev = math.sqrt(total_p * variance)
    scale = min(args.max_feature_rms / stddev, 1.0)
    offset = -mean * scale
    print("{0}\tunigram\t{1}\t{2}".format(num_features, offset, scale))
    num_features += 1

# length feature.  This feature is the length of the word, scaled
# down so that the rms does not exceed args.max_feature_rms.
# The format of the line is:
# <feature-index> length <scale>
# e.g.:
# 4 length 0.00518
if args.include_length_feature == 'true':
    feature_sumsq = 0.0
    for word_index, p in enumerate(unigram_probs):
        if word_index not in word_indexes_to_exclude:
            word = wordlist[word_index]
            feature_value = len(word)
            feature_sumsq += p * feature_value * feature_value
    rms = math.sqrt(feature_sumsq)
    print("{0}\tlength\t{1}".format(num_features, get_feature_scale(rms)))
    num_features += 1

# top-words features.  This is a feature that we assign to the top n most
# frequent words (e.g. the top 2000 most frequent words), *in addition* to any
# features they may get as a result of their written form.
# e.g. the line will look like:
# <feature-index> word <word> <scale>
# e.g.:
# 6 word of 1.0

# We need to remember which words are given these features to avoid printing
# the same feature later on when we process the n-gram type features; for
# instance, if we are including trigrams on characters, the 3-gram (BOS, a, EOS)
# would be the word "a", and if the word "a" got its own "word" feature due to
# the --top-word-features option, this would be a duplication.
# 'top_words' will be a set containing words (strings) that already had their
# own 'word' feature.
top_words = set()
if args.top_word_features > 0:
    # sorted_word_indexes is a sorted list of pairs (word_index, unigram_prob),
    # sorted from greatest to least unigram_prob.
    sorted_word_indexes = sorted(enumerate(unigram_probs),
                                 key=lambda x: x[1], reverse=True)
    num_top_words_printed = 0
    for word_index, unigram_prob in sorted_word_indexes:
        if word_index in word_indexes_to_exclude:
            continue
        word = wordlist[word_index]
        rms = math.sqrt(unigram_prob)
        print("{0}\tword\t{1}\t{2}".format(num_features, word, get_feature_scale(rms)))
        num_features += 1
        top_words.add(word)
        num_top_words_printed += 1
        if num_top_words_printed > args.top_word_features:
            break

# n-gram features.
# These are basically sub-strings of a word.  What we print in the feature
# file is things like:
#  <feat-index> (initial|final|match|word) <sub-string> <scale>
# e.g.
# 20 match tel 1.0
# where: 'match' means a match at any position in the word,
#        'final' means a match that must be word-final (and may or may
#                not be word-initial),
#        'initial' means a match that must be word-initial (and may or
#                may not be word-final)
#        'word' means a match that is both word-initial and word-final.
# For a given word, the feature value if there is a match will be the number of
# matches times the feature scale.

#  'ngram_feats' is a dict mapping a pair (match_type, match)
#  to a pair (feat_freq, expected_feat_sumsq), where:
#
#   match_type (a string) is one of: 'match', 'final', 'initial', 'word',
#           describing the match type, as explained above.
#   match (a string) is the string that we are matching, e.g. 'ing'.
#   feat_freq (a float) is the  sum over all words which the feature
#     appears in, of the probability of that word.
#   expected_feat_sumsq (a float) is the sum over all words of the probability
#    of that word, times the square of the number of times the feature
#    appears there.

# if you index ngram_feats with a key that wasn't present, you'll get the
# tuple (0.0, 0.0).
ngram_feats = defaultdict(lambda: (0.0, 0.0))

for (word_index, unigram_prob) in enumerate(unigram_probs):
    if word_index in word_indexes_to_exclude:
        continue
    word = wordlist[word_index]

    for pos in range(len(word) + 1):  # +1 for EOW 'this_word_feats' is a dict
        # from pairs (match_type, match) as defined above, to the count of the
        # number of times the feature was matched in this word (this count can
        # only be >1 if match_type == 'match').
        this_word_feats = defaultdict(int)

        for order in range(args.min_ngram_order, args.max_ngram_order + 1):
            start = pos - order + 1
            end = pos + 1

            if start < -1:
                continue

            if start < 0 and end > len(word):
                match_type = 'word'
                start = 0
                end = len(word)
            elif start < 0:
                match_type = 'initial'
                start = 0
            elif end > len(word):
                match_type = 'final'
                end = len(word)
            else:
                match_type = 'match'
            if start >= end:
                continue

            match = word[start:end]
            this_word_feats[(match_type, match)] += 1
        for (match_type, match), count in this_word_feats.items():
            (feat_freq, expected_feat_sumsq) = ngram_feats[(match_type, match)]
            ngram_feats[(match_type, match)] = (feat_freq + unigram_prob,
                                                expected_feat_sumsq + unigram_prob * count * count)


for (match_type, match), (expected_feat_sum, expected_feat_sumsq) in sorted(ngram_feats.items()):
    if match_type == 'word' and match in top_words:
        continue  # avoid duplicate
    if expected_feat_sum < args.min_frequency:
        continue  # very infrequent features are excluded via this mechanism.
    rms = math.sqrt(expected_feat_sumsq)
    print("{0}\t{1}\t{2}\t{3}".format(
        num_features, match_type, match, get_feature_scale(rms)))
    num_features += 1


print(sys.argv[0] + ": chose {0} features.".format(num_features), file=sys.stderr)
