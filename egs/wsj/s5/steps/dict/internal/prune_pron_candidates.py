#!/usr/bin/env python

# Copyright 2018  Xiaohui Zhang
# Apache 2.0.

from __future__ import print_function
from collections import defaultdict
import argparse
import sys
import math

def GetArgs():
    parser = argparse.ArgumentParser(
        description = "Prune pronunciation candidates based on soft-counts from lattice-alignment"
        "outputs, and a reference lexicon. Basically, for each word we sort all pronunciation"
        "cadidates according to their soft-counts, and then select the top variant-counts-ratio * N candidates"
        "(For words in the reference lexicon, N = # pron variants given by the reference"
        "lexicon; For oov words, N = avg. # pron variants per word in the reference lexicon).",
        epilog = "See steps/dict/learn_lexicon_greedy.sh for example")

    parser.add_argument("--variant-counts-ratio", type = float, default = "3.0",
                        help = "A user-specified ratio parameter which determines how many"
                        "pronunciation candidates we want to keep for each word at most.")
    parser.add_argument("pron_stats", metavar = "<pron-stats>", type = str,
                        help = "File containing soft-counts of pronounciation candidates; "
                        "each line must be <soft-counts> <word> <phones>")
    parser.add_argument("lexicon_phonetic_decoding", metavar = "<lexicon-phonetic-decoding>", type = str,
                        help = "Lexicon containing pronunciation candidates from phonetic decoding."
                        "each line must be <word> <phones>")
    parser.add_argument("lexiconp_g2p", metavar = "<lexiconp-g2p>", type = str,
                        help = "Lexicon with probabilities for pronunciation candidates from G2P."
                        "each line must be <prob> <word> <phones>")
    parser.add_argument("ref_lexicon", metavar = "<ref-lexicon>", type = str,
                        help = "Reference lexicon file, where we obtain # pron variants for"
                        "each word, based on which we prune the pron candidates."
                        "Each line must be <word> <phones>")
    parser.add_argument("lexicon_phonetic_decoding_pruned", metavar = "<lexicon-phonetic-decoding-pruned>", type = str,
                        help = "Output lexicon containing pronunciation candidates from phonetic decoding after pruning."
                        "each line must be <word> <phones>")
    parser.add_argument("lexicon_g2p_pruned", metavar = "<lexicon-g2p-pruned>", type = str,
                        help = "Output lexicon containing pronunciation candidates from G2P after pruning."
                        "each line must be <word> <phones>")

    print (' '.join(sys.argv), file=sys.stderr)

    args = parser.parse_args()
    args = CheckArgs(args)

    return args

def CheckArgs(args):
    print(args)
    args.pron_stats_handle = open(args.pron_stats)
    args.lexicon_phonetic_decoding_handle = open(args.lexicon_phonetic_decoding)
    args.lexiconp_g2p_handle = open(args.lexiconp_g2p)
    args.ref_lexicon_handle = open(args.ref_lexicon)
    args.lexicon_phonetic_decoding_pruned_handle = open(args.lexicon_phonetic_decoding_pruned, "w")
    args.lexicon_g2p_pruned_handle = open(args.lexicon_g2p_pruned, "w")
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

    return stats

def ReadLexicon(lexicon_handle):
    lexicon = defaultdict(set)
    for line in lexicon_handle.readlines():
        splits = line.strip().split()
        if len(splits) == 0:
            continue
        if len(splits) < 2:
            raise Exception('Invalid format of line ' + line
                                + ' in lexicon file.')
        word = splits[0]
        phones = ' '.join(splits[1:])
        lexicon[word].add(phones)
    return lexicon

def ReadLexiconp(lexiconp_handle):
    lexicon = defaultdict(set)
    pron_probs = defaultdict(float)
    for line in lexiconp_handle.readlines():
        splits = line.strip().split()
        if len(splits) == 0:
            continue
        if len(splits) < 3:
            raise Exception('Invalid format of line ' + line
                                + ' in lexicon file.')
        word = splits[1]
        prob = float(splits[0])
        phones = ' '.join(splits[2:])
        pron_probs[(word, phones)] = prob
        lexicon[word].add(phones)
    return lexicon, pron_probs

def PruneProns(args, stats, ref_lexicon, lexicon_phonetic_decoding, lexicon_g2p, lexicon_g2p_probs):
    # For those pron candidates from lexicon_phonetic_decoding/g2p which don't
    # have stats, we append them to the "stats" dict, with a zero count.
    for word, entry in stats.iteritems():
        prons_with_stats = set()
        for (pron, count) in entry:
            prons_with_stats.add(pron)
        for pron in lexicon_g2p[word]:
            if pron not in prons_with_stats:
                entry.append((pron, lexicon_g2p_probs[(word, pron)]-1.0))
        entry.sort(key=lambda x: x[1])
    
    # Compute the average # pron variants counts per word in the reference lexicon.
    num_words_ref = 0
    num_prons_ref = 0
    for word, prons in ref_lexicon.iteritems():
        num_words_ref += 1
        num_prons_ref += len(prons)
    avg_variant_counts_ref = round(float(num_prons_ref) / float(num_words_ref))
    for word, entry in stats.iteritems():
        if word in ref_lexicon:
            variant_counts = args.variant_counts_ratio * len(ref_lexicon[word])
        else:
            variant_counts = args.variant_counts_ratio * avg_variant_counts_ref
        num_variants = 0
        count = 0.0
        while num_variants < variant_counts:
            try:
                pron, count = entry.pop()
                if word in ref_lexicon and pron in ref_lexicon[word]:
                    continue
                if pron in lexicon_phonetic_decoding[word]:
                    num_variants += 1
                    print('{0} {1}'.format(word, pron), file=args.lexicon_phonetic_decoding_pruned_handle)
                if pron in lexicon_g2p[word]:
                    num_variants += 1
                    print('{0} {1}'.format(word, pron), file=args.lexicon_g2p_pruned_handle)
            except IndexError:
                break

def Main():
    args = GetArgs()
    ref_lexicon = ReadLexicon(args.ref_lexicon_handle)
    lexicon_phonetic_decoding = ReadLexicon(args.lexicon_phonetic_decoding_handle)
    lexicon_g2p, lexicon_g2p_probs = ReadLexiconp(args.lexiconp_g2p_handle)
    stats = ReadStats(args.pron_stats_handle)

    PruneProns(args, stats, ref_lexicon, lexicon_phonetic_decoding, lexicon_g2p, lexicon_g2p_probs)

if __name__ == "__main__":
    Main()
