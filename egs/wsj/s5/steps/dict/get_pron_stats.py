#!/usr/bin/env python

# Copyright 2016  Xiaohui Zhang
#           2016  Vimal Manohar
# Apache 2.0.

from __future__ import print_function
from collections import defaultdict
import argparse
import sys

def GetArgs():
    parser = argparse.ArgumentParser(
        description = "Accumulate statistics from lattice-alignment outputs for lexicon"
        "learning. The inputs are a file containing arc level information from lattice-align-words,"
        "and a map which maps word-position-dependent phones to word-position-independent phones"
        "(output from steps/cleanup/debug_lexicon.txt). The output contains accumulated soft-counts"
        "of pronunciations",
        epilog = "cat exp/tri3_lex_0.4_work/lats/arc_info_sym.*.txt \\|"
        "  steps/dict/get_pron_stats.py - exp/tri3_lex_0.4_work/phone_decode/phone_map.txt \\"
        "  exp/tri3_lex_0.4_work/lats/pron_stats.txt"
        "See steps/dict/learn_lexicon_greedy.sh for examples in detail.")

    parser.add_argument("arc_info_file", metavar = "<arc-info-file>", type = str,
                        help = "Input file containing per arc statistics; "
                        "each line must be <counts> <word> <phones>")
    parser.add_argument("phone_map", metavar = "<phone-map>", type = str,
                        help = "An input phone map used to remove word boundary markers from phones;"
                        "generated in steps/cleanup/debug_lexicon.sh")
    parser.add_argument("stats_file", metavar = "<stats_file>", type = str,
                        help = "Write accumulated statitistics to this file;"
                        "each line is <count> <word> <phones>")

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


def GetStatsFromArcInfo(arc_info_file_handle, phone_map_handle):
    prons = defaultdict(set)
    # need to map the phones to remove word boundary markers.
    phone_map = {}
    stats_unmapped = {} 
    stats = {} 
    for line in phone_map_handle.readlines():
        splits = line.strip().split()
        phone_map[splits[0]] = splits[1]

    for line in arc_info_file_handle.readlines():
        splits = line.strip().split()
        if (len(splits) == 0):
            continue
        if (len(splits) < 6):
            raise Exception('Invalid format of line ' + line
                                + ' in arc_info_file')
        word = splits[4]
        count = float(splits[3])
        phones = " ".join(splits[5:])        
        prons[word].add(phones)
        stats_unmapped[(word, phones)] = stats_unmapped.get((word, phones), 0) + count
     
    for word_pron, count in stats_unmapped.items():
        phones_unmapped = word_pron[1].split()
        phones = [phone_map[phone] for phone in phones_unmapped]
        stats[(word_pron[0], " ".join(phones))] = count
    return stats

def WriteStats(stats, file_handle):
    for word_pron, count in stats.items():
        print('{2} {0} {1}'.format(word_pron[0], word_pron[1], count),
              file=file_handle)
    file_handle.close()

def Main():
    args = GetArgs()
    stats = GetStatsFromArcInfo(args.arc_info_file_handle, args.phone_map_handle)
    WriteStats(stats, args.stats_file_handle)

if __name__ == "__main__":
    Main()
