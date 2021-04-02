#!/usr/bin/env python

# Copyright 2018   Xiaohui Zhang
# Apache 2.0

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
    parser = argparse.ArgumentParser(
        description = "Accumulate statistics from per arc lattice statitics"
        "for lexicon learning",
        epilog = "See steps/dict/learn_lexicon_greedy.sh for example")

    parser.add_argument("--set-sum-to-one", type = str, default = True,
                        action = StrToBoolAction, choices = ["true", "false"],
                        help = "If normalize posteriors such that the sum of "
                        "pronunciation posteriors of a word in an utterance is 1.")
    parser.add_argument("arc_info_file", metavar = "<arc-info-file>", type = str,
                        help = "File containing per arc statistics; "
                        "each line must be <utt-id> <word> <start-frame> <duration> <posterior>"
                        "<phones-with-word-boundary-markers>")
    parser.add_argument("phone_map", metavar = "<phone-map>", type = str,
                        help = "An input phone map used to remove word boundary markers from phones;"
                        "generated in steps/cleanup/debug_lexicon.sh")
    parser.add_argument("stats_file", metavar = "<out-stats-file>", type = str,
                        help = "Write accumulated statitistics to this file"
                        "each line is <utt-id> <word> <start-frame> <posterior>"
                        "<phones-without-word-boundary-markers>")

    print (' '.join(sys.argv), file=sys.stderr)

    args = parser.parse_args()
    args = CheckArgs(args)

    return args

def CheckArgs(args):
    if args.arc_info_file == "-":
        args.arc_info_file_handle = sys.stdin
    else:
        args.arc_info_file_handle = open(args.arc_info_file)
    
    args.phone_map_handle = open(args.phone_map)

    if args.stats_file == "-":
        args.stats_file_handle = sys.stdout
    else:
        args.stats_file_handle = open(args.stats_file, "w")

    return args

def Main():
    args = GetArgs()

    lexicon = defaultdict(list)
    prons = defaultdict(list)
    start_frames = {}
    stats = defaultdict(lambda : defaultdict(float))
    sum_tot = defaultdict(float)

    phone_map = {}
    for line in args.phone_map_handle.readlines():
        splits = line.strip().split()
        phone_map[splits[0]] = splits[1]

    for line in args.arc_info_file_handle.readlines():
        splits = line.strip().split()

        if (len(splits) == 0):
            continue

        if (len(splits) < 6):
            raise Exception('Invalid format of line ' + line
                                + ' in ' + args.arc_info_file)

        utt = splits[0]
        start_frame = int(splits[1])
        word = splits[4]
        count = float(splits[3])
        phones_unmapped = splits[5:]   
        phones = [phone_map[phone] for phone in phones_unmapped]
        phones = ' '.join(phones)
        overlap = False
        if word == '<eps>':
            continue
        if (word, utt) not in start_frames:
            start_frames[(word, utt)] = start_frame

        if (word, utt) in stats:
            stats[word, utt][phones] = stats[word, utt].get(phones, 0) + count
        else:
            stats[(word, utt)][phones] = count
        sum_tot[(word, utt)] += count

        if phones not in prons[word]:
            prons[word].append(phones)

    for (word, utt) in stats:
       count_sum = 0.0
       counts = dict()
       for phones in stats[(word, utt)]:
           count = stats[(word, utt)][phones]
           count_sum += count
           counts[phones] = count
       # By default we normalize the pron posteriors of each word in each utterance,
       # so that they sum up exactly to one. If a word occurs two times in a utterance,
       # the effect of this operation is to average the posteriors of these two occurences
       # so that there's only one "equivalent occurence" of this word in the utterance.
       # However, this case should be extremely rare if the utterances are already
       # short sub-utterances produced by steps/dict/internal/get_subsegments.py
       for phones in stats[(word, utt)]:
           count = counts[phones] / count_sum
           print(word, utt, start_frames[(word, utt)], count, phones, file=args.stats_file_handle)
       # # Diagnostics info implying incomplete arc_info or multiple occurences of a word in a utterance:
       # if count_sum < 0.9 or count_sum > 1.1:
       #    print(word, utt, start_frame, count_sum, stats[word, utt], file=sys.stderr)

    args.stats_file_handle.close()

if __name__ == "__main__":
    Main()
