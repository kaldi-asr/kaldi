#!/usr/bin/env python

# Copyright 2016  Vimal Manohar
#           2016  Xiaohui Zhang
# Apache 2.0.

# we're using python 3.x style print but want it to work in python 2.x,
from __future__ import print_function
from collections import defaultdict
import argparse
import sys

class StrToBoolAction(argparse.Action):
    """ A custom action to convert bools from shell format i.e., true/false
        to python format i.e., True/False """
    def __call__(self, parser, namespace, values, option_string=None):
        if values == "true":
            setattr(namespace, self.dest, True)
        elif values == "false":
            setattr(namespace, self.dest, False)
        else:
            raise Exception("Unknown value {0} for --{1}".format(values, self.dest))

def GetArgs():
    parser = argparse.ArgumentParser(description = "Converts pronunciation statistics (from phonetic decoding or g2p) "
                                     "into a lexicon for. We prune the pronunciations "
                                     "based on a provided stats file, and optionally filter out entries which are present "
                                     "in a filter lexicon.",
                                     epilog = "e.g. steps/dict/prons_to_lexicon.py --min-prob=0.4 \\"
                                     "--filter-lexicon=exp/tri3_lex_0.4_work/phone_decode/filter_lexicon.txt \\"
                                     "exp/tri3_lex_0.4_work/phone_decode/prons.txt \\"
                                     "exp/tri3_lex_0.4_work/lexicon_phone_decoding.txt"
                                     "See steps/dict/learn_lexicon_greedy.sh for examples in detail.")

    parser.add_argument("--set-sum-to-one", type = str, default = False,
                        action = StrToBoolAction, choices = ["true", "false"],
                        help = "If normalize lexicon such that the sum of "
                        "probabilities is 1.")
    parser.add_argument("--set-max-to-one", type = str, default = True,
                        action = StrToBoolAction, choices = ["true", "false"],
                        help = "If normalize lexicon such that the max "
                        "probability is 1.")
    parser.add_argument("--top-N", type = int, default = 0,
                        help = "If non-zero, we just take the top N pronunciations (according to stats/pron-probs) for each word.")
    parser.add_argument("--min-prob", type = float, default = 0.1,
                        help = "Remove pronunciation with probabilities less "
                        "than this value after normalization.")
    parser.add_argument("--filter-lexicon", metavar='<filter-lexicon>', type = str, default = '',
                        help = "Exclude entries in this filter lexicon from the output lexicon."
                        "each line must be <word> <phones>")
    parser.add_argument("stats_file", metavar='<stats-file>', type = str,
                        help = "Input lexicon file containing pronunciation statistics/probs in the first column."
                        "each line must be <counts> <word> <phones>")
    parser.add_argument("out_lexicon", metavar='<out-lexicon>', type = str,
                        help = "Output lexicon.")

    print (' '.join(sys.argv), file = sys.stderr)

    args = parser.parse_args()
    args = CheckArgs(args)

    return args

def CheckArgs(args):
    if args.stats_file == "-":
        args.stats_file_handle = sys.stdin
    else:
        args.stats_file_handle = open(args.stats_file)

    if args.filter_lexicon is not '':
        if args.filter_lexicon == "-":
            args.filter_lexicon_handle = sys.stdout
        else:
            args.filter_lexicon_handle = open(args.filter_lexicon)
    
    if args.out_lexicon == "-":
        args.out_lexicon_handle = sys.stdout
    else:
        args.out_lexicon_handle = open(args.out_lexicon, "w")

    if args.set_max_to_one == args.set_sum_to_one:
        raise Exception("Cannot have both "
            "set-max-to-one and set-sum-to-one as true or false.")

    return args

def ReadStats(args):
    lexicon = {}
    word_count = {}
    for line in args.stats_file_handle:
        splits = line.strip().split()
        if len(splits) < 3:
            continue

        word = splits[1]
        count = float(splits[0])
        phones = ' '.join(splits[2:])

        lexicon[(word, phones)] = lexicon.get((word, phones), 0) + count
        word_count[word] = word_count.get(word, 0) + count

    return [lexicon, word_count]

def ReadLexicon(lexicon_file_handle):
    lexicon = set()
    if lexicon_file_handle:
        for line in lexicon_file_handle.readlines():
            splits = line.strip().split()
            if len(splits) == 0:
                continue
            if len(splits) < 2:
                raise Exception('Invalid format of line ' + line
                                    + ' in lexicon file.')
            word = splits[0]
            phones = ' '.join(splits[1:])
            lexicon.add((word, phones))
    return lexicon

def ConvertWordCountsToProbs(args, lexicon, word_count):
    word_probs = {}
    for entry, count in lexicon.iteritems():
        word = entry[0]
        phones = entry[1]
        prob = float(count) / float(word_count[word])
        if word in word_probs:
            word_probs[word].append((phones, prob))
        else:
            word_probs[word] = [(phones, prob)]

    return word_probs

def ConvertWordProbsToLexicon(word_probs):
    lexicon = {}
    for word, entry in word_probs.iteritems():
        for x in entry:
            lexicon[(word, x[0])] = lexicon.get((word,x[0]), 0) + x[1]
    return lexicon

def NormalizeLexicon(lexicon, set_max_to_one = True,
                     set_sum_to_one = False, min_prob = 0):
    word_probs = {}
    for entry, prob in lexicon.iteritems():
        t = word_probs.get(entry[0], (0,0))
        word_probs[entry[0]] = (t[0] + prob, max(t[1], prob))

    for entry, prob in lexicon.iteritems():
        if set_max_to_one:
            prob = prob / word_probs[entry[0]][1]
        elif set_sum_to_one:
            prob = prob / word_probs[entry[0]][0]
        if prob < min_prob:
            prob = 0
        lexicon[entry] = prob

def TakeTopN(lexicon, top_N):
    lexicon_reshaped = defaultdict(list) 
    lexicon_pruned = {}
    for entry, prob in lexicon.iteritems():
        lexicon_reshaped[entry[0]].append([entry[1], prob])
    for word in lexicon_reshaped:
        prons = lexicon_reshaped[word]
        sorted_prons = sorted(prons, reverse=True, key=lambda prons: prons[1])
        for i in range(len(sorted_prons)):
            if i >= top_N:
                lexicon[(word, sorted_prons[i][0])] = 0
        
def WriteLexicon(args, lexicon, filter_lexicon):
    words = set()
    num_removed = 0
    num_filtered = 0
    for entry, prob in lexicon.iteritems():
        if prob == 0:
            num_removed += 1
            continue
        if entry in filter_lexicon:
            num_filtered += 1
            continue
        words.add(entry[0])
        print("{0} {1}".format(entry[0], entry[1]),
                file = args.out_lexicon_handle)
    print ("Before pruning, the total num. pronunciations is: {}".format(len(lexicon)), file=sys.stderr)
    print ("Removed {0} pronunciations by setting min_prob {1}".format(num_removed, args.min_prob), file=sys.stderr)
    print ("Filtered out {} pronunciations in the filter lexicon.".format(num_filtered), file=sys.stderr)
    num_prons_from_phone_decoding = len(lexicon) - num_removed - num_filtered
    print ("Num. pronunciations in the output lexicon, which solely come from phone decoding"
           "is {0}. num. words is {1}".format(num_prons_from_phone_decoding, len(words)), file=sys.stderr)

def Main():
    args = GetArgs()

    [lexicon, word_count] = ReadStats(args)

    word_probs = ConvertWordCountsToProbs(args, lexicon, word_count)

    lexicon = ConvertWordProbsToLexicon(word_probs)
    filter_lexicon = set()
    if args.filter_lexicon is not '':
        filter_lexicon = ReadLexicon(args.filter_lexicon_handle)
    if args.top_N > 0:
        TakeTopN(lexicon, args.top_N)
    else:
        NormalizeLexicon(lexicon, set_max_to_one = args.set_max_to_one,
                         set_sum_to_one = args.set_sum_to_one,
                         min_prob = args.min_prob)
    WriteLexicon(args, lexicon, filter_lexicon)
    args.out_lexicon_handle.close()

if __name__ == "__main__":
    Main()
