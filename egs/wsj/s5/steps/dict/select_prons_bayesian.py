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
import math
import numpy

def GetArgs():
    parser = argparse.ArgumentParser(description = "Use a Bayesian framework to select"
                                     "pronunciation candidates from three sources: reference lexicon"
                                     ", G2P lexicon and phone-decoding lexicon. The inputs are a word-stats file,"
                                     "a pron-stats file, and three source lexicons (ref/G2P/phone-decoding)."
                                     "We assume the pronunciations for each word follow a Categorical distribution"
                                     "with Dirichlet priors. Thus, with user-specified prior counts (parameterized by"
                                     "prior-mean and prior-count-tot) and observed counts from the pron-stats file, "
                                     "we can compute posterior for each pron, and select candidates with highest"
                                     "posteriors, until we hit user-specified variants-prob-mass/counts thresholds."
                                     "The outputs are: a file specifiying posteriors of all candidate (pron_posteriors),"
                                     "a learned lexicon for words out of the ref. vocab (learned_lexicon_oov),"
                                     "and a lexicon_edits file containing suggested modifications of prons, for"
                                     "words within the ref. vocab (ref_lexicon_edits)."
                                     epilog = "See steps/dict/learn_lexicon.sh for example.")
    parser.add_argument("--prior-mean", type = str, default = "0,0,0",
                        help = "Mean of priors (summing up to 1) assigned to three exclusive n"
                        "pronunciatio sources: reference lexicon, g2p, and phone decoding. We "
                        "recommend setting a larger prior mean for the reference lexicon, e.g. '0.6,0.2,0.2'"
    parser.add_argument("--prior-counts-tot", type = float, default = 15.0)
                        help = "Total amount of prior counts we add to all pronunciation candidates of"
                        "each word. By timing it with the prior mean of a source, and then dividing"
                        "by the number of candidates (for a word) from this source, we get the"
                        "prior counts we actually add to each candidate."
    parser.add_argument("--variants-prob-mass", type = float, default = 0.7,
                        help = "For each word, we pick up candidates (from all three sources)"
                        "with highest posteriors until the total prob mass hit this amount."
    parser.add_argument("--variants-prob-mass2", type = float, default = 0.9,
                        help = "For each word, after the total prob mass of selected candidates "
                        "hit variants-prob-mass, we continue to pick up reference candidates"
                        "with highest posteriors until the total prob mass hit this amount."
    parser.add_argument("--variants-counts", type = int, default = 1,
                        help = "Generate upto this many variants of prons for each word out"
                        "of the ref. lexicon.")
    parser.add_argument("silence_file", metavar = "<silphone-file>", type = str,
                        help = "File containing a list of silence phones.")
    parser.add_argument("pron_stats_file", metavar = "<stats-file>", type = str,
                        help = "File containing pronunciation statistics from lattice alignment; "
                        "each line must be <count> <word> <phones>.")
    parser.add_argument("word_counts_file", metavar = "<counts-file>", type = str,
                        help = "File containing word counts in acoustic training data; "
                        "each line must be <word> <count>.")
    parser.add_argument("ref_lexicon", metavar = "<reference-lexicon>", type = str,
                        help = "The reference lexicon (most probably hand-derived).")
    parser.add_argument("g2p_lexicon", metavar = "<g2p-expanded-lexicon>", type = str,
                        help = "Pronouciations from G2P results.")
    parser.add_argument("phone_decoding_lexicon", metavar = "<prons-in-acoustic-evidence>", type = str,
                        help = "Pronouciations from phone decoding results.")
    parser.add_argument("pron_posteriors", metavar = "<pron-posteriors>", type = str,
                        help = "File containing posteriors of all candidate prons for each word,"
                        "based on which we select prons to construct the learned lexicon.")
    parser.add_argument("learned_lexicon_oov", metavar = "<learned-lexicon-oov>", type = str,
                        help = "Learned lexicon for words out of the ref. vocab.")
    parser.add_argument("ref_lexicon_edits", metavar = "<lexicon-edits>", type = str,
                        help = "File containing human-readable & editable pronounciation info (and the"
                        "accept/reject decision made by our algorithm) for those words in ref. vocab," 
                        "to which any change has been recommended. i.e. at least one ref. candidate "
                        "prons was rejected, or at least one candidate pron from G2P/phone-decoding is accepted."
                        "This file possibly indicates wrongly derived ref. prons / text normalization issues.")

    print (' '.join(sys.argv), file=sys.stderr)

    args = parser.parse_args()
    args = CheckArgs(args)

    return args

def CheckArgs(args):
    args.silence_file_handle = open(args.silence_file)
    if args.pron_stats_file == "-":
        args.pron_stats_file_handle = sys.stdin
    else:
        args.pron_stats_file_handle = open(args.pron_stats_file)
    args.word_counts_file_handle = open(args.word_counts_file)
    args.ref_lexicon_handle = open(args.ref_lexicon)
    args.g2p_lexicon_handle = open(args.g2p_lexicon)
    args.phone_decoding_handle = open(args.phone_decoding_lexicon)
    args.pron_posteriors_handle = open(args.pron_posteriors, "w")
    args.learned_lexicon_oov_handle = open(args.learned_lexicon_oov, "w")
    args.ref_lexicon_edits_handle = open(args.ref_lexicon_edits, "w")
    
    prior_mean = args.prior_mean.strip().split(',')
    if len(prior_mean) is not 3:
        raise Exception('Invalid Dirichlet prior mean ' + args.prior_mean)
    for i in range(0,3):
        if float(prior_mean[i]) <= 0 or float(prior_mean[i]) >= 1:
            raise Exception('Dirichlet prior mean', prior_mean[i], 'is invalid, it must be between 0 and 1.')
    args.prior_mean = [float(prior_mean[0]), float(prior_mean[1]), float(prior_mean[2])]

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

def ReadLexicon(args, lexicon_file_handle, counts):
    # we're skipping any word not in counts (not seen in training data),
    # cause we're only learning prons for words who have acoustic examples.
    lexicon = defaultdict(set)
    for line in lexicon_file_handle.readlines():
        splits = line.strip().split()
        if len(splits) == 0:
            continue
        if len(splits) < 2:
            raise Exception('Invalid format of line ' + line
                                + ' in lexicon file.')
        word = splits[0]
        if word not in counts:
            continue
        try:
            prob = float(splits[1])
            phones = ' '.join(splits[2:])
        except ValueError:
            prob = 1
            phones = ' '.join(splits[1:])
        lexicon[word].add(phones)
    return lexicon

def FilterPhoneDecodingLexicon(args, phone_decoding_lexicon, stats):
    # We want to remove all candidates which contains silence phones
    silphones = set()
    for line in args.silence_file_handle:
        silphones.add(line.strip())
    rejected_candidates = set()
    for word, prons in phone_decoding_lexicon.iteritems():
        for pron in prons:
            for phone in pron.split():
                if phone in silphones:
                   if (word, pron) in stats:
                       count = stats[(word, pron)]
                       del stats[(word, pron)]
                   else:
                       count = 0
                   rejected_candidates.add((word, pron))
                   print("WARNING: removing the candidate pronunciation from phone-decoding:", word, ":",
                         pron, "whose soft-count from lattice-alignment is ", count, ", cause it contains at least one silence phone." )
                   break
    for word, pron in rejected_candidates:
        phone_decoding_lexicon[word].remove(pron)
    return phone_decoding_lexicon, stats

def ComputePriorCounts(args, counts, ref_lexicon, g2p_lexicon, phone_decoding_lexicon):
    prior_counts = defaultdict(list)
    # In case one source is absent for a word, we set zero prior to this source, 
    # and then re-normalize the prior mean parameters s.t. they sum up to one.
    for word in counts:
        prior_mean = [args.prior_mean[0], args.prior_mean[1], args.prior_mean[2]]
        if word not in ref_lexicon:
            prior_mean[0] = 0
        if word not in g2p_lexicon:
            prior_mean[1] = 0
        if word not in phone_decoding_lexicon:
            prior_mean[2] = 0
        prior_mean_sum = sum(prior_mean)
        try:
            prior_mean = [t / prior_mean_sum for t in prior_mean] 
        except ZeroDivisionError:
            print("WARNING: word", word, "appears in train_counts but not in any lexicon.")
        prior_counts[word] = [t * args.prior_counts_tot for t in prior_mean] 
    return prior_counts

def ComputePosteriors(args, stats, ref_lexicon, g2p_lexicon, phone_decoding_lexicon, prior_counts):
    posteriors = defaultdict(list) # This dict stores a list of (pronunciation, posterior)
    # pairs for each word, where the posteriors are normalized soft counts. Before normalization,
    # The soft-counts were augmented by a user-specified prior count, according the source 
    # (ref/G2P/phone-decoding) of this pronunciation.

    for word, prons in ref_lexicon.iteritems():
        for pron in prons:
            posteriors[word].append((pron, prior_counts[word][0] / len(ref_lexicon[word]) + stats.get((word, pron), 0)))

    for word, prons in g2p_lexicon.iteritems():
        for pron in prons:
            posteriors[word].append((pron, prior_counts[word][1] / len(g2p_lexicon[word]) + stats.get((word, pron), 0)))

    for word, prons in phone_decoding_lexicon.iteritems():
        for pron in prons:
            posteriors[word].append((pron, prior_counts[word][2] / len(phone_decoding_lexicon[word]) + stats.get((word, pron), 0)))

    print ("---------------------------------------------------------------------------------------------------")
    print ("Total num. words is:", len(posteriors))
    print (sum(len(ref_lexicon[i]) for i in ref_lexicon), "candidate prons came from the reference lexicon;", 
           sum(len(g2p_lexicon[i]) for i in g2p_lexicon), "came from G2P;",
           sum(len(phone_decoding_lexicon[i]) for i in phone_decoding_lexicon), "came from phone-decoding.")
    print ("---------------------------------------------------------------------------------------------------")

    # Normalize the augmented soft counts to get posteriors.
    count_sum = defaultdict(float) # This dict stores the pronunciation which has 
    # the sum of augmented soft counts for each word.
    
    for word in posteriors:
        # each entry is a pair: (prounciation, count)
        count_sum[word] = sum([entry[1] for entry in posteriors[word]])
        if count_sum[word] == 0:
            print(word, posteriors[word], prior_counts[word] )
    
    for word, entry in posteriors.iteritems():
        new_entry = []
        for pron, count in entry:      
            post = count / count_sum[word]
            new_entry.append((pron, post))
            source = 'R'
            if word in g2p_lexicon and pron in g2p_lexicon[word]:
                source = 'G'
            elif word in phone_decoding_lexicon and pron in phone_decoding_lexicon[word]:
                source = 'P'
            print(word, source, "%3.2f" % post, pron, file=args.pron_posteriors_handle)
        del entry[:]
        entry.extend(sorted(new_entry, key=lambda new_entry: new_entry[1]))
    return posteriors

def SelectPronsBayesian(args, counts, posteriors, ref_lexicon, g2p_lexicon, phone_decoding_lexicon):
    reference_selected = 0
    g2p_selected = 0
    phone_decoding_selected = 0
    low_max_post_words = set()
    learned_lexicon = defaultdict(set)

    for word, entry in posteriors.iteritems():
        num_variants = 0
        post_tot = 0.0
        variants_counts = args.variants_counts
        variants_prob_mass = args.variants_prob_mass
        if word in ref_lexicon:
            # the variants count of the current word's prons in the ref lexicon.
            variants_counts_ref = len(ref_lexicon[word])
            # For words who don't appear in acoustic training data at all, we simply accept all ref prons.
            # For words in ref. vocab, we set the max num. variants 
            if counts.get(word, 0) > 0:
                variants_counts = math.ceil(1.5 * variants_counts_ref)
            else:
                variants_counts = variants_counts_ref
                variants_prob_mass = 1.0
        last_post = 0.0
        while (num_variants < variants_counts and post_tot < variants_prob_mass) or (len(entry) > 0 and entry[-1][1] == last_post):
            try:
                pron, post = entry.pop()
                last_post = post
            except IndexError:
                break
            post_tot += post
            learned_lexicon[word].add(pron)
            num_variants += 1
            if word in ref_lexicon and pron in ref_lexicon[word]:
                reference_selected += 1
            elif word in g2p_lexicon and pron in g2p_lexicon[word]:
                g2p_selected += 1
            else:
                phone_decoding_selected += 1

        while (num_variants < variants_counts and post_tot < args.variants_prob_mass2):
            try:
                pron, post = entry.pop()
            except IndexError:
                break
            if word in ref_lexicon and pron in ref_lexicon[word]:
                post_tot += post
                learned_lexicon[word].add(pron)
                num_variants += 1
                reference_selected += 1

    num_prons_tot = reference_selected + g2p_selected + phone_decoding_selected
    print("---------------------------------------------------------------------------------------------------")
    print ("Num. words in the learned lexicon:", len(learned_lexicon), "; num. selected prons:", num_prons_tot)
    print (reference_selected, "selected prons came from reference candidate prons;", g2p_selected, "came from G2P candidate prons;", phone_decoding_selected, "came from phone-decoding candidate prons.")
    return learned_lexicon

def WriteEditsAndSummary(args, learned_lexicon, ref_lexicon, phone_decoding_lexicon, g2p_lexicon, counts, stats):
    # Note that learned_lexicon and ref_lexicon are dicts of sets of prons, while the other two lexicons are sets of (word, pron) pairs.
    thr = 3
    words = [defaultdict(set) for i in range(4)] # "words" contains four bins, where we
    # classify each word into, according to whether it's count > thr,
    # and whether it's OOVs w.r.t the reference lexicon.

    src = {}
    print("# Note: This file contains pronunciation info for words who have candidate prons from G2P/phone-decoding accepted in the learned lexicon."
          ", sorted by their counts in acoustic training data, "
          ,file=args.ref_lexicon_edits_handle)
    print("# 1st Col: source of the candidate pron: G(2P) / P(hone-decoding) / R(eference)."
          ,file=args.ref_lexicon_edits_handle)
    print("# 2nd Col: soft counts from lattice-alignment (not augmented by prior-counts)."
          ,file=args.ref_lexicon_edits_handle)
    print("# 3rd Col: accepted or not in the learned lexicon (Y/N).", file=args.ref_lexicon_edits_handle)
    print("# 4th Col: the pronunciation cadidate.", file=args.ref_lexicon_edits_handle)
    
    # words which are to be printed into the edits file.
    words_to_edit = [] 
    for word in learned_lexicon:
        count = counts.get(word, 0)
        flags = ['0' for i in range(3)] # "flags" contains three binary indicators, 
        # indicating where this word's pronunciations come from.
        for pron in learned_lexicon[word]:
            if word in phone_decoding_lexicon and pron in phone_decoding_lexicon[word]:
                flags[0] = '1'
                src[(word, pron)] = 'P'
            if word in ref_lexicon and pron in ref_lexicon[word]:
                flags[1] = '1'
                src[(word, pron)] = 'R'
            if word in g2p_lexicon and pron in g2p_lexicon[word]:
                flags[2] = '1'
                src[(word, pron)] = 'G'
        if word in ref_lexicon:
            all_ref_prons_accepted = True
            for pron in ref_lexicon[word]:
                if pron not in learned_lexicon[word]:
                    all_ref_prons_accepted = False
                    break
            if not all_ref_prons_accepted or flags[0] == '1' or flags[2] == '1':
                words_to_edit.append((word, counts[word]))
            if count > thr:
                words[0][flags[0] + flags[1] + flags[2]].add(word)
            else:
                words[1][flags[0] + flags[1] + flags[2]].add(word)
        else:
            if count > thr: 
                words[2][flags[0] + flags[2]].add(word)
            else:
                words[3][flags[0] + flags[2]].add(word)

    words_to_edit_sorted = sorted(words_to_edit, key=lambda entry: entry[1], reverse=True)
    for word, count in words_to_edit_sorted:
        print("------------",word, "%2.1f" % count, "--------------", file=args.ref_lexicon_edits_handle)
        for pron in learned_lexicon[word]:
            print(src[(word, pron)], ' | Y | ', "%2.1f | " % stats.get((word, pron), 0), pron, 
                  file=args.ref_lexicon_edits_handle)
        for pron in ref_lexicon[word]:
            if pron not in learned_lexicon[word]:
                print('R', ' | N | ', "%2.1f | " % stats.get((word, pron), 0), pron, 
                      file=args.ref_lexicon_edits_handle)
    print("Here are the words whose reference pron candidates were all declined", words[0]['100'])
    print("-------------------------------------------------Summary------------------------------------------")
    print("In the learned lexicon, out of those", len(ref_lexicon), "words from the vocab of the reference lexicon:") 
    print("  For those words whose counts in the training text > ", thr, ":") 
    print("    ", len(words[0]['111']) + len(words[0]['110']) + len(words[0]['011']), " words' selected prons came from the reference lexicon, G2P/phone-decoding.")
    print("    ", len(words[0]['101']) + len(words[0]['001']) + len(words[0]['100']), " words' selected prons come from G2P/phone-decoding-generated.") 
    print("    ", len(words[0]['010']), " words' selected prons came from the reference lexicon only.") 
    print("  For those words whose counts in the training text <=", thr, ":") 
    print("    ", len(words[1]['111']) + len(words[1]['110']) + len(words[1]['011']), " words' selected prons came from the reference lexicon, G2P/phone-decoding.")
    print("    ", len(words[1]['101']) + len(words[1]['001']) + len(words[1]['100']), " words' selected prons come from G2P/phone-decoding-generated.") 
    print("    ", len(words[1]['010']), " words' selected prons came from the reference lexicon only.") 
    print("---------------------------------------------------------------------------------------------------")
    print("  In the learned lexicon, out of those", len(learned_lexicon) - len(ref_lexicon), "OOV words (w.r.t the reference lexicon):")
    print("  For those words whose counts in the training text >", thr, ":") 
    print("    ", len(words[2]['11']), " words' selected prons came from G2P and phone-decoding.")
    print("    ", len(words[2]['10']), " words' selected prons came from phone decoding only.") 
    print("    ", len(words[2]['01']), " words' selected prons came from G2P only.")
    print("  For those words whose counts in the training text <=", thr, ":") 
    print("    ", len(words[3]['11']), " words' selected prons came from G2P and phone-decoding.")
    print("    ", len(words[3]['10']), " words' selected prons came from phone decoding only.") 
    print("    ", len(words[3]['01']), " words' selected prons came from G2P only.")

def WriteLearnedLexiconOov(learned_lexicon, ref_lexicon, file_handle):
    for word, prons in learned_lexicon.iteritems():
        if word not in ref_lexicon:
            for pron in prons:
                print('{0} {1}'.format(word, pron), file=file_handle)
    file_handle.close()

def Main():
    args = GetArgs()

    # Read in three lexicon sources, word counts, and pron stats.
    counts = ReadWordCounts(args.word_counts_file_handle)
    ref_lexicon = ReadLexicon(args, args.ref_lexicon_handle, counts)
    g2p_lexicon = ReadLexicon(args, args.g2p_lexicon_handle, counts)
    phone_decoding_lexicon =  ReadLexicon(args, args.phone_decoding_handle, counts)
    stats = ReadPronStats(args.pron_stats_file_handle)
    phone_decoding_lexicon, stats = FilterPhoneDecodingLexicon(args, phone_decoding_lexicon, stats)
   
    # Compute prior counts
    prior_counts = ComputePriorCounts(args, counts, ref_lexicon, g2p_lexicon, phone_decoding_lexicon)
    # Compute posteriors, and then select prons to construct the learned lexicon.
    posteriors = ComputePosteriors(args, stats, ref_lexicon, g2p_lexicon, phone_decoding_lexicon, prior_counts)

    # Select prons to construct the learned lexicon.
    learned_lexicon = SelectPronsBayesian(args, counts, posteriors, ref_lexicon, g2p_lexicon, phone_decoding_lexicon)
    
    # Write the learned prons for words out of the ref. vocab into learned_lexicon_oov.
    WriteLearnedLexiconOov(learned_lexicon, ref_lexicon, args.learned_lexicon_oov_handle)
    # Edits will be printed into ref_lexicon_edits, and the summary will be printed into stdout.
    WriteEditsAndSummary(args, learned_lexicon, ref_lexicon, phone_decoding_lexicon, g2p_lexicon, counts, stats)

if __name__ == "__main__":
    Main()
