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

def GetArgs():
    parser = argparse.ArgumentParser(description = "Accumulate statistics from arc_info"
                                     "for lexicon learning",
                                     epilog = "See steps/dict/debug_lexicon.sh for example")

    parser.add_argument("arc_info_file", metavar = "<arc-info-file>", type = str,
                        help = "File containing per arc statistics; "
                        "each line must be <counts> <word> <phones>")
    parser.add_argument("stats_file", metavar = "<out-stats-file>", type = str,
                        help = "Write accumulated statitistics to this file")

    print (' '.join(sys.argv))

    args = parser.parse_args()
    args = CheckArgs(args)

    return args

def CheckArgs(args):
    if args.arc_info_file == "-":
        args.arc_info_file_handle = sys.stdin
    else:
        args.arc_info_file_handle = open(args.arc_info_file)

    if args.stats_file == "-":
        args.stats_file_handle = sys.stdout
    else:
        args.stats_file_handle = open(args.stats_file, "w")

    return args

def Main():
    args = GetArgs()

    lexicon = {}

    for line in args.arc_info_file_handle.readlines():
        splits = line.strip().split()

        if (len(splits) == 0):
            continue

        if (len(splits) < 3):
            raise Exception('Invalid format of line ' + line
                                + ' in ' + args.arc_info_file)

        word = splits[1]
        count = float(splits[0])
        phones = " ".join(splits[2:])

        lexicon[(word, phones)] = lexicon.get((word, phones), 0) + count

    for entry, count in lexicon.iteritems():
        print('{0} {1} {2}'.format(count, entry[0], entry[1]),
                                    file=args.stats_file_handle)

    args.stats_file_handle.close()

if __name__ == "__main__":
    Main()
