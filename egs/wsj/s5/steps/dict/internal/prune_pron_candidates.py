#!/usr/bin/env python

# Copyright 2016  Xiaohui Zhang
# Apache 2.0.

from __future__ import print_function
from collections import defaultdict
import argparse
import sys
import math

def GetArgs():
    parser = argparse.ArgumentParser(description = "Prune pronunciation candidates based on soft-counts from lattice-alignment"
                                     "outputs, and a reference lexicon. Basically, for each word we sort all pronunciation"
                                     "cadidates according to their soft-counts, and then select the top r * N candidates"
                                     "(For words in the reference lexicon, N = # pron variants given by the reference"
                                     "lexicon; For oov words, N = avg. # pron variants per word in the reference lexicon)."
                                     "r is a user-specified constant, like 2.",
                                     epilog = "See steps/dict/learn_lexicon.sh for example")

    parser.add_argument("--r", type = float, default = "2.0",
                        help = "a user-specified ratio parameter which determines how many"
                        "pronunciation candidates we want to keep for each word.")
    parser.add_argument("pron_stats", metavar = "<pron-stats>", type = str,
                        help = "File containing soft-counts of all pronounciation candidates; "
                        "each line must be <soft-counts> <word> <phones>")
    parser.add_argument("ref_lexicon", metavar = "<ref-lexicon>", type = str,
                        help = "Reference lexicon file, where we obtain # pron variants for"
                        "each word, based on which we prune the pron candidates."
                        "Each line must be <word> <phones>")
    parser.add_argument("pruned_prons", metavar = "<pruned-prons>", type = str,
                        help = "An output file in lexicon format, which contains prons we want to" 
                        "prune off from the pron_stats file.")

    print (' '.join(sys.argv), file=sys.stderr)

    args = parser.parse_args()
    args = CheckArgs(args)

    return args

def CheckArgs(args):
    args.pron_stats_handle = open(args.pron_stats)
    args.ref_lexicon_handle = open(args.ref_lexicon)
    if args.pruned_prons == "-":
        args.pruned_prons_handle = sys.stdout
    else:
        args.pruned_prons_handle = open(args.pruned_prons, "w")
    return args

def ReadStats(pron_stats_handle):
    stats = defaultdict(list)
    for line in pron_stats_handle.readlines():
        splits = line.strip().split()
        if len(splits) == 0:
            continue
        if len(splits) < 2:
            raise Exception('Invalid format of line ' + line
                                + ' in stats file.')
        count = float(splits[0])
        word = splits[1]
        phones = ' '.join(splits[2:])
        stats[word].append((phones, count))

    for word, entry in stats.iteritems():
        entry.sort(key=lambda x: x[1])
    return stats

def ReadLexicon(ref_lexicon_handle):
    ref_lexicon = defaultdict(set)
    for line in ref_lexicon_handle.readlines():
        splits = line.strip().split()
        if len(splits) == 0:
            continue
        if len(splits) < 2:
            raise Exception('Invalid format of line ' + line
                                + ' in lexicon file.')
        word = splits[0]
        phones = ' '.join(splits[1:])
        ref_lexicon[word].add(phones)
    return ref_lexicon

def PruneProns(args, stats, ref_lexicon):
    # Compute the average # pron variants counts per word in the reference lexicon.
    num_words_ref = 0
    num_prons_ref = 0
    for word, prons in ref_lexicon.iteritems():
        num_words_ref += 1
        num_prons_ref += len(prons)
    avg_variants_counts_ref = math.ceil(float(num_prons_ref) / float(num_words_ref))

    for word, entry in stats.iteritems():
        if word in ref_lexicon:
            variants_counts = args.r * len(ref_lexicon[word])
        else:
            variants_counts = args.r * avg_variants_counts_ref
        num_variants = 0
        while num_variants < variants_counts:
            try:
                pron, prob = entry.pop()
                if word not in ref_lexicon or pron not in ref_lexicon[word]:
                    num_variants += 1
            except IndexError:
                break
        
    for word, entry in stats.iteritems():
        for pron, prob in entry:
            if word not in ref_lexicon or pron not in ref_lexicon[word]:
                print('{0} {1}'.format(word, pron), file=args.pruned_prons_handle)

def Main():
    args = GetArgs()
    ref_lexicon = ReadLexicon(args.ref_lexicon_handle)
    stats = ReadStats(args.pron_stats_handle)
    PruneProns(args, stats, ref_lexicon)

if __name__ == "__main__":
    Main()
