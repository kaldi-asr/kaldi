#!/usr/bin/env python

# Copyright 2016  Xiaohui Zhang
# Apache 2.0.

from __future__ import print_function
from collections import defaultdict
import os
import argparse
import sys
import warnings
import copy
import imp
import ast
import numpy

def GetArgs():
    parser = argparse.ArgumentParser(description = "Use a Bayesian framework to select"
                                     "pronounciations from three sources: reference lexicon (most probably hand-derived)"
                                     ", G2P lexicon and phone-decoding lexicon. The inputs are a word stats file,"
                                     "a pron stats file, and three source lexicons (ref/G2P/phone-decoding)."
                                     "We assume the pronounciations for each word follow a Categorical distribution"
                                     "with Dirichlet priors. Thus, with user-specified prior counts and observed counts"
                                     "in the pron stats file, we can compute posterior for each pron, and select"
                                     "the best prons thresholded by user-specified variants-mass/counts to"
                                     "construct the learned lexicon. The outputs are a learned lexicon (out_lexicon),"
                                     "posteriors of all candidate prons (pron_posteriors), and word-pron info of words"
                                     "whose reference candidate prons were rejected. (diagnostic_info)",
                                     epilog = "See steps/dict/expand_lex_learned.sh for example.")

    parser.add_argument("--alpha", type = str, default = "0-0-0",
                        help = "prior counts (Dirichlet prior hyperparameters) "
                        "corresponding to the three pronounciation sources: reference lexicon; G2P; phone decoding.")
    parser.add_argument("--variants-mass", type = float, default = 0.8,
                        help = "Generate so many variants of prons to produce this amount of "
                        "prob mass, for each word in the learned lexicon.")
    parser.add_argument("--variants-counts", type = int, default = 1,
                        help = "Generate upto this many variants of prons for each word in"
                        "the learned lexicon.")
    parser.add_argument("pron_stats_file", metavar = "<stats-file>", type = str,
                        help = "File containing pronounciation statistics; "
                        "each line must be <count> <word> <phones>.")
    parser.add_argument("word_counts_file", metavar = "<counts-file>", type = str,
                        help = "File containing word counts; "
                        "each line must be <word> <count>.")
    parser.add_argument("reference_lexicon", metavar = "<reference-lexicon>", type = str,
                        help = "The reference lexicon (most probably hand-derived).")
    parser.add_argument("g2p_lexicon", metavar = "<g2p-expanded-lexicon>", type = str,
                        help = "Pronouciations from G2P results.")
    parser.add_argument("phone_decoding_lexicon", metavar = "<prons-in-acoustic-evidence>", type = str,
                        help = "Pronouciations from phone decoding results.")
    parser.add_argument("out_lexicon", metavar = "<out-lexicon>", type = str,
                        help = "Write the learned lexicon to this file.")
    parser.add_argument("pron_posteriors", metavar = "<pron-posteriors>", type = str,
                        help = "File containing posteriors of all candidate prons for each word,"
                        "based on which we select prons to construct the learned lexicon.")
    parser.add_argument("diagnostic_info", metavar = "<diagnostic-info>", type = str,
                        help = "File containing word-pron info, for those words whose pron candidates"
                        "from the reference lexicon were rejected, and candidates from phone-decoding/G2P"
                        "were selected, in out_lexicon. It's likely the prons from the reference lexicon"
                        "are wrongly derived, or there're text normalization issues.")

    print (' '.join(sys.argv), file=sys.stderr)

    args = parser.parse_args()
    args = CheckArgs(args)

    return args

def CheckArgs(args):
    if args.pron_stats_file == "-":
        args.pron_stats_file_handle = sys.stdin
    else:
        args.pron_stats_file_handle = open(args.pron_stats_file)
    args.word_counts_file_handle = open(args.word_counts_file)
    args.reference_lexicon_handle = open(args.reference_lexicon)
    args.g2p_lexicon_handle = open(args.g2p_lexicon)
    args.phone_decoding_handle = open(args.phone_decoding_lexicon)
    args.out_lexicon_handle = open(args.out_lexicon, "w")
    args.pron_posteriors_handle = open(args.pron_posteriors, "w")
    args.diagnostic_info_handle = open(args.diagnostic_info, "w")
    
    alpha = args.alpha.strip().split('-')
    if len(alpha) is not 3:
        raise Exception('Invalid hyperparameter for the Dirichlet priors ' + args.alpha)
    args.alpha = [float(alpha[0]), float(alpha[1]), float(alpha[2])]

    return args

def ReadPronStats(pron_stats_file_handle):
    stats = {}
    for line in pron_stats_file_handle.readlines():
        splits = line.strip().split()
        if len(splits) == 0:
            continue
        if len(splits) < 2:
            raise Exception('Invalid format of line ' + line
                                + ' in stats file.')
        count = float(splits[0])
        word = splits[1]
        phones = ' '.join(splits[2:])
        stats[(word, phones)] = count
    return stats

def ReadWordCounts(word_counts_file_handle):
    counts = {}
    for line in word_counts_file_handle.readlines():
        splits = line.strip().split()
        if len(splits) < 2:
            raise Exception('Invalid format of line ' + line
                                + ' in counts file.')
        word = splits[0]
        count = int(splits[1])
        counts[word] = count
    return counts

def ReadLexicon(args, lexicon_file_handle):
    lexicon = set()
    for line in lexicon_file_handle.readlines():
        splits = line.strip().split()
        if len(splits) == 0:
            continue
        if len(splits) < 2:
            raise Exception('Invalid format of line ' + line
                                + ' in lexicon file.')
        word = splits[0]
        try:
            prob = float(splits[1])
            phones = ' '.join(splits[2:])
        except ValueError:
            prob = 1
            phones = ' '.join(splits[1:])
        lexicon.add((word, phones))
    return lexicon

def ComputePosteriors(args, stats, reference_lexicon, g2p_lexicon, phone_decoding_lexicon):
    posteriors = defaultdict(list) # This dict stores a list of (pronounciation, posterior)
    # pairs for each word, where the posteriors are normalized soft counts. Before normalization,
    # The soft-counts were augmented by a user-specified prior count, according the source 
    # (ref/G2P/phone-decoding) of this pronounciation.

    seen_words = set()
    for word_pron, count in stats.iteritems():
        word = word_pron[0]
        pron = word_pron[1]
        seen_words.add(word)
        if word_pron in reference_lexicon:
            posteriors[word].append((pron, float(count) + args.alpha[0])) 
        elif (word, pron) in g2p_lexicon:
            posteriors[word].append((pron, float(count) + args.alpha[1])) 
        else:
            posteriors[word].append((pron, float(count) + args.alpha[2]))
    num_reference_unseen = 0
    for word, pron in reference_lexicon:
        if (word, pron) not in stats:
            num_reference_unseen += 1
            posteriors[word].append((pron, args.alpha[0]))

    num_g2p_unseen = 0
    for word, pron in g2p_lexicon:
        if (word, pron) not in stats:
            num_g2p_unseen += 1
            posteriors[word].append((pron, args.alpha[1]))

    num_phone_decoding_unseen = 0
    for word, pron in phone_decoding_lexicon:
        if (word, pron) not in stats:
            num_phone_decoding_unseen += 1
            posteriors[word].append((pron, args.alpha[2]))

    num_reference_seen = len(reference_lexicon) - num_reference_unseen
    num_g2p_seen = len(g2p_lexicon) - num_g2p_unseen
    num_phone_decoding_seen = len(phone_decoding_lexicon) - num_phone_decoding_unseen

    num_prons_tot = len(stats) + num_reference_unseen + num_g2p_unseen
    print("---------------------------------------------------------------------------------------------------")
    print ("Total num. words is:", len(posteriors), ", total num. candidate prons:", num_prons_tot)
    print (len(reference_lexicon), "candidate prons came from the reference lexicon;", len(g2p_lexicon), "came from G2P;", len(phone_decoding_lexicon), "came from phone-decoding.")
    print("---------------------------------------------------------------------------------------------------")
    print ("Num. words seen in the acoustic evidence (have non-zero soft counts in lattices):", len(seen_words), "; num. seen prons:", len(stats))
    print (num_reference_seen, "seen prons came from the reference candidate prons;", num_g2p_seen, "came from G2P candidate prons;", num_phone_decoding_seen,"came from phone-decoding candidate prons.")

    # Normalize the augmented soft counts to get posteriors.
    count_sum = {} # This dict stores the pronounciation which has 
    # the sum of augmented soft counts for each word.
    
    for word, entry_list in posteriors.iteritems():
        # each entry is a pair: (prounciation, count)
        for entry in entry_list:
            count = entry[1]
            count_sum[word] = count_sum.get(word, 0) + count
    
    for word, entry in posteriors.iteritems():
        new_entry = []
        for pron, count in entry:      
            post = count / count_sum[word]
            new_entry.append((pron, post))
            source = 'R'
            if (word, pron) in g2p_lexicon:
                source = 'G'
            elif (word, pron) in phone_decoding_lexicon:
                source = 'P'
            print(word, source, "%3.2f" % post, pron, file=args.pron_posteriors_handle)
        del entry[:]
        entry.extend(sorted(new_entry, key=lambda new_entry: new_entry[1]))
        
    return posteriors

def SelectPronsBayesian(args, posteriors, reference_lexicon, g2p_lexicon, phone_decoding_lexicon):
    reference_selected = 0
    g2p_selected = 0
    phone_decoding_selected = 0
    low_max_post_words = set()
    out_lexicon = defaultdict(set)
    for word, entry in posteriors.iteritems():
        num_variants = 0
        post_tot = 0.0
        while num_variants < args.variants_counts:
            try:
                pron, post = entry.pop() 
            except IndexError:
                break
            post_tot += post
            if post_tot > args.variants_mass and len(out_lexicon[word]) > 0:
                break
            out_lexicon[word].add(pron)
            num_variants += 1
            if (word, pron) in reference_lexicon:
                reference_selected += 1
            elif (word, pron) in g2p_lexicon:
                g2p_selected += 1
            else:
                phone_decoding_selected += 1

    num_prons_tot = reference_selected + g2p_selected + phone_decoding_selected
    print("---------------------------------------------------------------------------------------------------")
    print ("Num. words in the learned lexicon:", len(out_lexicon), "; num. selected prons:", num_prons_tot)
    print (reference_selected, "selected prons came from reference candidate prons;", g2p_selected, "came from G2P candidate prons;", phone_decoding_selected, "came from phone-decoding candidate prons.")
    print("---------------------------------------------------------------------------------------------------")
    return out_lexicon

def PrintInfo(args, out_lexicon, reference_lexicon, phone_decoding_lexicon, g2p_lexicon, counts, stats):
    thr = 3
    words = [defaultdict(set) for i in range(4)] # "words" contains four bins, where we
    # classify each word into, according to whether it's count > thr,
    # and whether it's OOVs w.r.t the reference lexicon.

    reference_lexicon_reshaped = defaultdict(set)
    for entry in reference_lexicon:
        reference_lexicon_reshaped[entry[0]].add(entry[1])
    src = {}
    print("# Note: This file only contains pronounciation info for 'bad words' (whose reference candidate prons were all rejected)"
          ", sorted by a badness score, which is max(count of selected prons) - max(count of rejected prons), "
          ,file=args.diagnostic_info_handle)
    print("# 1st Col: source of the candidate pron: G(2P) / P(hone-decoding) / R(eference)."
          ,file=args.diagnostic_info_handle)
    print("# 2nd Col: soft counts from lattice-alignment (not augmented by prior-counts)."
          ,file=args.diagnostic_info_handle)
    print("# 3rd Col: selected or not (Y/N).", file=args.diagnostic_info_handle)
    print("# 4th Col: the pronounciation cadidate.", file=args.diagnostic_info_handle)
    bad_words = [] 
    for word in out_lexicon:
        count = counts.get(word, 0)
        flags = ['0' for i in range(3)] # "flags" contains three binary indicators, 
        # indicating where this word's pronounciations come from.
        for pron in out_lexicon[word]:
            if (word, pron) in phone_decoding_lexicon:
                flags[0] = '1'
                src[(word, pron)] = 'P'
            if (word, pron) in reference_lexicon:
                flags[1] = '1'
                src[(word, pron)] = 'R'
            if (word, pron) in g2p_lexicon:
                flags[2] = '1'
                src[(word, pron)] = 'G'
        if word in reference_lexicon_reshaped:
            if flags[1] == '0':
                max_count_selected = 0
                max_count_rejected = 0
                for pron in out_lexicon[word]:
                    max_count_selected = max(max_count_selected, stats.get((word, pron), 0))
                for pron in reference_lexicon_reshaped[word]:
                    max_count_rejected = max(max_count_rejected, stats.get((word, pron), 0))
                bad_words.append((word, max_count_selected - max_count_rejected))
            if count > thr:
                words[0][flags[0] + flags[1] + flags[2]].add(word)
            else:
                words[1][flags[0] + flags[1] + flags[2]].add(word)
        else:
            if count > thr: 
                words[2][flags[0] + flags[2]].add(word)
            else:
                words[3][flags[0] + flags[2]].add(word)

    bad_words_sorted = sorted(bad_words, key=lambda entry: entry[1], reverse=True)
    for word, badness in bad_words_sorted:
        print("------------",word, "%2.1f" % badness, "--------------", file=args.diagnostic_info_handle)
        for pron in out_lexicon[word]:
            print(src[(word, pron)], ' | Y | ', "%2.1f | " % stats.get((word, pron), 0), pron, 
                  file=args.diagnostic_info_handle)
        for pron in reference_lexicon_reshaped[word]:
            print('R', ' | N | ', "%2.1f | " % stats.get((word, pron), 0), pron, 
                  file=args.diagnostic_info_handle)
    a = len(words[0]['101']) + len(words[0]['100']) + len(words[0]['001'])
    b = len(words[1]['101']) + len(words[1]['100']) + len(words[1]['001'])
    print("-------------------------------------------------Summary------------------------------------------")
    print("In the learned lexicon, out of those", len(reference_lexicon_reshaped), "words from the vocab of the reference lexicon:") 
    print("  For those words whose counts in the training text > ", thr, ":") 
    print("    ", len(words[0]['111']), " words' selected prons came from the reference lexicon, G2P and phone-decoding.")
    print("    ", a, " words' selected prons come from G2P/phone-decoding-generated.") 
    print("    ", len(words[0]['010']), " words' selected prons came from the reference lexicon only.") 
    print("  For those words whose counts in the training text <=", thr, ":") 
    print("    ", len(words[1]['111']), " words' selected prons came from the reference lexicon, G2P and phone-decoding.")
    print("    ", b, " words' selected prons come from G2P/phone-decoding-generated.") 
    print("    ", len(words[1]['010']), " words' selected prons came from the reference lexicon only.") 
    print("Please see file", args.diagnostic_info, "for detailed info about those", "%d + %d = %d" %(a,b,a+b), 
          "in-vocab words whose candidate prons from the reference lexicon were all rejected.")            
    print("---------------------------------------------------------------------------------------------------")
    print("  In the learned lexicon, out of those", len(out_lexicon) - len(reference_lexicon_reshaped), "OOV words (w.r.t the reference lexicon):")
    print("  For those words whose counts in the training text >", thr, ":") 
    print("    ", len(words[2]['11']), " words' selected prons came from G2P and phone-decoding.")
    print("    ", len(words[2]['10']), " words' selected prons came from phone decoding only.") 
    print("    ", len(words[2]['01']), " words' selected prons came from G2P only.")
    print("  For those words whose counts in the training text <=", thr, ":") 
    print("    ", len(words[3]['11']), " words' selected prons came from G2P and phone-decoding.")
    print("    ", len(words[3]['10']), " words' selected prons came from phone decoding only.") 
    print("    ", len(words[3]['01']), " words' selected prons came from G2P only.")

def WriteOutLexicon(out_lexicon, file_handle):
    for word, prons in out_lexicon.iteritems():
        for pron in prons:
            print('{0} {1}'.format(word, pron), file=file_handle)
    file_handle.close()

def Main():
    args = GetArgs()

    # Read in three lexicon sources, word counts, and pron stats.
    reference_lexicon = ReadLexicon(args, args.reference_lexicon_handle)
    g2p_lexicon = ReadLexicon(args, args.g2p_lexicon_handle)
    phone_decoding_lexicon =  ReadLexicon(args, args.phone_decoding_handle)
    stats = ReadPronStats(args.pron_stats_file_handle)
    counts = ReadWordCounts(args.word_counts_file_handle)
    
    # Augmente the pron stats by prior counts, compute posteriors, and then select prons to construct the learned lexicon.
    posteriors = ComputePosteriors(args, stats, reference_lexicon, g2p_lexicon, phone_decoding_lexicon)
    out_lexicon = SelectPronsBayesian(args, posteriors, reference_lexicon, g2p_lexicon, phone_decoding_lexicon)
    WriteOutLexicon(out_lexicon, args.out_lexicon_handle)

    # Print diagnostic info. The summary will only be printed to stdout.
    PrintInfo(args, out_lexicon, reference_lexicon, phone_decoding_lexicon, g2p_lexicon, counts, stats)

if __name__ == "__main__":
    Main()
