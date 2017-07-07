#!/usr/bin/env python3

import os
import argparse
import sys

parser = argparse.ArgumentParser(description="This script gets the unigram probabilities of words.",
                                 epilog="E.g. " + sys.argv[0] + " --vocab-file=data/rnnlm/vocab/words.txt "
                                        "--data-weights-file=exp/rnnlm/data_weights.txt data/rnnlm/data "
                                        "> exp/rnnlm/unigram_probs.txt",
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument("--vocab-file", type=str, default='', required=True,
                    help="Specify the vocab file.")
parser.add_argument("--data-weights-file", type=str, default='', required=True,
                    help="Specify the data-weights file.")
parser.add_argument("--smooth-unigram-counts", type=float, default=1.0,
                    help="Specify the constant for smoothing. We will add "
                         "(smooth_unigram_counts * num_words_with_non_zero_counts / vocab_size) "
                         "to every unigram counts.")
parser.add_argument("data_dir",
                    help="Directory in which to look for data")

args = parser.parse_args()

SPECIAL_SYMBOLS = ["<eps>", "<s>", "<brk>"]


# get the name with txt and counts file path for all data sources except dev
# return a dict with key is the name of data_source,
#                    value is a tuple (txt_file_path, counts_file_path)
def get_all_data_sources_except_dev(data_dir):
    data_sources = {}
    for f in os.listdir(data_dir):
        full_path = data_dir + "/" + f
        if f == 'dev.txt' or f == 'dev.counts' or os.path.isdir(full_path):
            continue
        if f.endswith(".txt"):
            name = f[0:-4]
            if name in data_sources:
                data_sources[name] = (full_path, data_sources[name][1])
            else:
                data_sources[name] = (full_path, None)
        elif f.endswith(".counts"):
            name = f[0:-7]
            if name in data_sources:
                data_sources[name] = (data_sources[name][0], full_path)
            else:
                data_sources[name] = (None, full_path)
        else:
            sys.exit(sys.argv[0] + ": Text directory should not contain files with suffixes "
                     "other than .txt or .counts: " + f)

    for name, (txt_file, counts_file) in data_sources.items():
        if txt_file is None or counts_file is None:
            sys.exit(sys.argv[0] + ": Missing .txt or .counts file for data source: " + name)

    return data_sources


# read the data-weights for data_sources from weights_file
# return a dict with key is name of a data source,
#                    value is a tuple (repeated_times_per_epoch, weight)
def read_data_weights(weights_file, data_sources):
    data_weights = {}
    with open(weights_file, 'r', encoding="utf-8") as f:
        for line in f:
            fields = line.split()
            assert len(fields) == 3
            if fields[0] in data_weights:
                sys.exit(sys.argv[0] + ": duplicated data source({0}) specified in "
                                       "data-weights: {1}".format(fields[0], weights_file))
            data_weights[fields[0]] = (int(fields[1]), float(fields[2]))

    for name in data_sources.keys():
        if name not in data_weights:
            sys.exit(sys.argv[0] + ": Weight for data source '{0}' not set".format(name))

    return data_weights


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


# Get total (weighted) count for words from all data_sources
# return a list of counts indexed by word id.
def get_counts(data_sources, data_weights, vocab):
    counts = [0.0] * len(vocab)

    for name, (_, counts_file) in data_sources.items():
        weight = data_weights[name][0] * data_weights[name][1]
        if weight == 0.0:
            continue

        with open(counts_file, 'r', encoding="utf-8") as f:
            for line in f:
                fields = line.split()
                assert len(fields) == 2

                counts[vocab[fields[0]]] += weight * int(fields[1])

    return counts


# Smooth counts and get unigram probs for words
# return a list of probs indexed by word id.
def get_unigram_probs(vocab, counts, smooth_constant):
    special_symbol_ids = [vocab[x] for x in SPECIAL_SYMBOLS]
    vocab_size = len(vocab) - len(SPECIAL_SYMBOLS)
    num_words_with_non_zero_counts = 0
    for word_id, count in enumerate(counts):
        if word_id in special_symbol_ids:
            continue
        if counts[word_id] > 0:
            num_words_with_non_zero_counts += 1

    if num_words_with_non_zero_counts < vocab_size and smooth_constant == 0.0:
        sys.exit(sys.argv[0] + ": --smooth-unigram-counts should not be zero, "
                               "since there are words with zero-counts")

    smooth_count = smooth_constant * num_words_with_non_zero_counts / vocab_size

    total_counts = 0.0
    for word_id, count in enumerate(counts):
        if word_id in special_symbol_ids:
            continue
        counts[word_id] += smooth_count
        total_counts += counts[word_id]

    probs = []
    for count in counts:
        probs.append(count / total_counts)

    return probs


data_sources = get_all_data_sources_except_dev(args.data_dir)
data_weights = read_data_weights(args.data_weights_file, data_sources)
vocab = read_vocab(args.vocab_file)

counts = get_counts(data_sources, data_weights, vocab)
probs = get_unigram_probs(vocab, counts, args.smooth_unigram_counts)

for idx, p in enumerate(probs):
    print(idx, p)

print(sys.argv[0] + ": generated unigram probs.", file=sys.stderr)
