#!/usr/bin/env python

# we're using python 3.x style print but want it to work in python 2.x,
from __future__ import print_function
import os
import argparse
import sys
import warnings
import copy
import imp
import ast

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
    parser = argparse.ArgumentParser(description = "Converts pronunciation statistics into "
                                     "lexicon"
                                     "for lexicon learning",
                                     epilog = "See steps/dict/debug_lexicon.sh for example")

    parser.add_argument("--max-prons-weight", type = float, default = 0.9,
                        help = "Keep pronunciations accounting for this "
                        "fraction of the total count of a word")
    parser.add_argument("--set-sum-to-one", type = str, default = False,
                        action = StrToBoolAction, choices = ["true", "false"],
                        help = "If normalize lexicon such that the sum of "
                        "probability is 1")
    parser.add_argument("--set-max-to-one", type = str, default = True,
                        action = StrToBoolAction, choices = ["true", "false"],
                        help = "If normalize lexicon such that the max "
                        "probability is 1")
    parser.add_argument("--min-prob", type = float, default = 0.1,
                        help = "Remove pronunciation with probabilities less "
                        "than this value after normalization")
    parser.add_argument("--min-count", type = float, default = 0,
                        help = "Ignore stats below this count")
    parser.add_argument("stats_file", metavar='<stats-file>', type = str,
                        help = "File containing statistics; "
                        "each line must be <counts> <word> <phones>")
    parser.add_argument("out_lexicon", metavar='<out-lexicon>', type = str,
                        help = "Write output lexicon")

    print (' '.join(sys.argv), file = sys.stderr)

    args = parser.parse_args()
    args = CheckArgs(args)

    return args

def CheckArgs(args):
    if args.stats_file == "-":
        args.stats_file_handle = sys.stdin
    else:
        args.stats_file_handle = open(args.stats_file)

    if args.out_lexicon == "-":
        args.out_lexicon_handle = sys.stdout
    else:
        args.out_lexicon_handle = open(args.out_lexicon, "w")

    if args.set_max_to_one and args.set_sum_to_one:
        raise Exception("Cannot have both "
            "set-max-to-one and set-sum-to-one as true")

    return args

def ReadStats(args):
    lexicon = {}
    word_count = {}
    for line in args.stats_file_handle:
        splits = line.strip().split()
        if len(splits) == 0:
            continue
        if len(splits) < 3:
            raise Exception('Invalid format of line ' + line
                                + ' in ' + args.arc_info_file)

        word = splits[1]
        count = float(splits[0])
        phones = '\t'.join(splits[2:])

        lexicon[(word, phones)] = lexicon.get((word, phones), 0) + count
        word_count[word] = word_count.get(word, 0) + count

    return [lexicon, word_count]

def ConvertWordCountsToProbs(args, lexicon, word_count):
    word_probs = {}
    for entry, count in lexicon.iteritems():
        word = entry[0]
        phones = entry[1]
        if count < args.min_count:
            continue

        prob = float(count) / float(word_count[word])
        if word in word_probs:
            word_probs[word].append((phones, prob))
        else:
            word_probs[word] = [(phones, prob)]

    return word_probs

def ProcessWordProbs(args, word_probs):
    for word, entry in word_probs.iteritems():
        entry.sort(key = lambda x: x[1], reverse = True)

        cumulative_sum = 0
        new_entry = []
        for x in entry:
            cumulative_sum += x[1]
            if cumulative_sum > args.max_prons_weight:
                break
            new_entry.append((x[0], x[1]))
        del entry[:]
        entry.extend(new_entry)

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
        elif set_max_to_one:
            prob = prob / word_probs[entry[0]][0]
        if prob < min_prob:
            prob = 0
        lexicon[entry] = prob

def WriteLexicon(lexicon, file_handle):
    for entry, prob in lexicon.iteritems():
        if prob == 0:
            continue
        print("{0}\t{1}\t{2}".format(entry[0], prob, entry[1]),
                file = file_handle)


def Main():
    args = GetArgs()

    [lexicon, word_count] = ReadStats(args)

    word_probs = ConvertWordCountsToProbs(args, lexicon, word_count)
    ProcessWordProbs(args, word_probs)

    lexicon = ConvertWordProbsToLexicon(word_probs)
    NormalizeLexicon(lexicon, set_max_to_one = args.set_max_to_one,
                     set_sum_to_one = args.set_sum_to_one,
                     min_prob = args.min_prob)
    WriteLexicon(lexicon, args.out_lexicon_handle)
    args.out_lexicon_handle.close()

if __name__ == "__main__":
    Main()
