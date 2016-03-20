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
    parser = argparse.ArgumentParser(description = "Merges two lexicons while "
                                     "scaling the probabilities on them")

    parser.add_argument("--weights", type = str,
                        help = "A colon separated list of weights on the "
                        "probabilities of the lexicons. If a lexicon, does not "
                        "have probabilities then the the probability of a "
                        "pronunciation is taken to be 1")
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
    parser.add_argument("out_lexicon", metavar='<out-lexicon>', type = str,
                        help = "Write output lexicon")
    parser.add_argument("in_lexicon", metavar='<in-lexicon>', type = str,
                        nargs='+',
                        help = "Input lexicons to be combined")

    print (' '.join(sys.argv), file = sys.stderr)

    args = parser.parse_args()
    args = CheckArgs(args)

    return args

def CheckArgs(args):
    if args.out_lexicon == "-":
        args.out_lexicon_handle = sys.stdout
    else:
        args.out_lexicon_handle = open(args.out_lexicon, "w")

    if args.weights is not None:
        args.weights = [ float(x) for x in args.weights.strip().split(':') ]
    else:
        args.weights = [ 1 for x in args.in_lexicon ]

    if len(args.weights) != len(args.in_lexicon):
        raise Exception("The length of --weights must be the same as that of "
                        "the number of input lexicons to be combined")

    args.in_lexicon_handle = []
    for i in range(0, len(args.in_lexicon)):
        if args.in_lexicon[i] == "-":
            args.in_lexicon_handle.append(sys.stdin)
        else:
            args.in_lexicon_handle.append(open(args.in_lexicon[i]))

    if args.set_max_to_one and args.set_sum_to_one:
        raise Exception("Cannot have both "
            "set-max-to-one and set-sum-to-one as true")

    return args

def ReadLexicon(file_handle, weight = 1):
    lexicon = {}
    for line in file_handle.readlines():
        splits = line.strip().split()
        if len(splits) == 0:
            continue
        if len(splits) < 3:
            raise Exception('Invalid format of line ' + line
                                + ' in ' + args.arc_info_file)

        word = splits[0]
        try:
            prob = float(splits[1])
            phones = '\t'.join(splits[2:])
        except ValueError:
            prob = 1
            phones = '\t'.join(splits[1:])

        if (word, phones) in lexicon:
            raise Exception('Found duplicate ({0},{1}) in lexicon'.format(word, phones))

        lexicon[(word, phones)] = prob * weight
    return lexicon

def MergeLexicon(file_handle, lexicon, weight = 1):
    for line in file_handle.readlines():
        splits = line.strip().split()
        if len(splits) == 0:
            continue
        if len(splits) < 3:
            raise Exception('Invalid format of line ' + line
                                + ' in ' + args.arc_info_file)

        word = splits[0]
        try:
            prob = float(splits[1])
            phones = '\t'.join(splits[2:])
        except ValueError:
            prob = 1
            phones = '\t'.join(splits[1:])

        lexicon[(word, phones)] = lexicon.get((word, phones), 0) + prob * weight

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
            prob  = prob / word_probs[entry[0]][0]
        if prob < min_prob:
            prob = 0
        lexicon[entry] = prob

def WriteLexicon(lexicon, file_handle):
    for entry, prob in lexicon.iteritems():
        if prob > 0:
            print("{0}\t{1}\t{2}".format(entry[0], prob, entry[1]),
                    file = file_handle)


def Main():
    args = GetArgs()

    lexicon = ReadLexicon(args.in_lexicon_handle[0], args.weights[0])

    for i in range(1, len(args.in_lexicon_handle)):
        MergeLexicon(args.in_lexicon_handle[i], lexicon, args.weights[1])

    NormalizeLexicon(lexicon, set_max_to_one = args.set_max_to_one,
                     set_sum_to_one = args.set_sum_to_one,
                     min_prob = args.min_prob)
    WriteLexicon(lexicon, args.out_lexicon_handle)

    args.out_lexicon_handle.close()

if __name__ == "__main__":
    Main()

