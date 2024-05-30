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
        description = "Convert a learned lexicon produced by steps/dict/select_prons_greedy.sh"
        "into a lexicon for OOV words (w.r.t. ref. vocab) and a human editable lexicon-edit file."
        "for in-vocab words, and generate detailed summaries of the lexicon learning results"
        "The inputs are a learned lexicon, an arc-stats file, and three source lexicons "
        "(phonetic-decoding(PD)/G2P/ref). The outputs are: a learned lexicon for OOVs"
        "(learned_lexicon_oov), and a lexicon_edits file (ref_lexicon_edits) containing"
        "suggested modifications of prons, for in-vocab words.",
        epilog = "See steps/dict/learn_lexicon_greedy.sh for example.")
    parser.add_argument("arc_stats_file", metavar = "<arc-stats-file>", type = str,
                        help = "File containing word-pronunciation statistics obtained from lattices; "
                        "each line must be <word> <utt-id> <start-frame> <count> <phones>")
    parser.add_argument("word_counts_file", metavar = "<counts-file>", type = str,
                        help = "File containing word counts in acoustic training data; "
                        "each line must be <word> <count>.")
    parser.add_argument("ref_lexicon", metavar = "<reference-lexicon>", type = str,
                        help = "The reference lexicon (most probably hand-derived)."
                        "Each line must be <word> <phones>")
    parser.add_argument("g2p_lexicon", metavar = "<g2p-expanded-lexicon>", type = str,
                        help = "Candidate ronouciations from G2P results."
                        "Each line must be <word> <phones>")
    parser.add_argument("pd_lexicon", metavar = "<prons-in-acoustic-evidence>", type = str,
                        help = "Candidate ronouciations from phonetic decoding results."
                        "Each line must be <word> <phones>")
    parser.add_argument("learned_lexicon", metavar = "<learned-lexicon>", type = str,
                        help = "Learned lexicon."
                        "Each line must be <word> <phones>")
    parser.add_argument("learned_lexicon_oov", metavar = "<learned-lexicon-oov>", type = str,
                        help = "Output file which is the learned lexicon for words out of the ref. vocab.")
    parser.add_argument("ref_lexicon_edits", metavar = "<lexicon-edits>", type = str,
                        help = "Output file containing human-readable & editable pronounciation info (and the"
                        "accept/reject decision made by our algorithm) for those words in ref. vocab," 
                        "to which any change has been recommended. The info for each word is like:" 
                        "------------ an 4086.0 --------------"
                        "R  | Y |  2401.6 |  AH N"
                        "R  | Y |  640.8 |  AE N"
                        "P  | Y |  1035.5 |  IH N"
                        "R(ef), P(hone-decoding) represents the pronunciation source"
                        "Y/N means the recommended decision of including this pron or not"
                        "and the numbers are soft counts accumulated from lattice-align-word outputs. "
                        "See the function WriteEditsAndSummary for more details.")
 
    print (' '.join(sys.argv), file=sys.stderr)

    args = parser.parse_args()
    args = CheckArgs(args)

    return args

def CheckArgs(args):
    if args.arc_stats_file == "-":
        args.arc_stats_file_handle = sys.stdin
    else:
        args.arc_stats_file_handle = open(args.arc_stats_file)
    args.word_counts_file_handle = open(args.word_counts_file)
    args.ref_lexicon_handle = open(args.ref_lexicon)
    args.g2p_lexicon_handle = open(args.g2p_lexicon)
    args.pd_lexicon_handle = open(args.pd_lexicon)
    args.learned_lexicon_handle = open(args.learned_lexicon)
    args.learned_lexicon_oov_handle = open(args.learned_lexicon_oov, "w")
    args.ref_lexicon_edits_handle = open(args.ref_lexicon_edits, "w")
    
    return args

def ReadArcStats(arc_stats_file_handle):
    stats = defaultdict(lambda : defaultdict(dict))
    stats_summed = defaultdict(float)
    for line in arc_stats_file_handle.readlines():
        splits = line.strip().split()

        if (len(splits) == 0):
            continue

        if (len(splits) < 5):
            raise Exception('Invalid format of line ' + line
                                + ' in ' + arc_stats_file)
        utt = splits[1]
        start_frame = int(splits[2])
        word = splits[0]
        count = float(splits[3])
        phones = splits[4:]
        phones = ' '.join(phones)
        stats[word][(utt, start_frame)][phones] = count
        stats_summed[(word, phones)] += count
    return stats, stats_summed

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
        phones = ' '.join(splits[1:])
        lexicon[word].add(phones)
    return lexicon

def WriteEditsAndSummary(args, learned_lexicon, ref_lexicon, pd_lexicon, g2p_lexicon, counts, stats, stats_summed):
    # Note that learned_lexicon and ref_lexicon are dicts of sets of prons, while the other two lexicons are sets of (word, pron) pairs.
    threshold = 2
    words = [defaultdict(set) for i in range(4)] # "words" contains four bins, where we
    # classify each word into, according to whether it's count > threshold,
    # and whether it's OOVs w.r.t the reference lexicon.

    src = {}
    print("# Note: This file contains pronunciation info for words who have candidate "
          "prons from G2P/phonetic-decoding accepted in the learned lexicon"
          ", sorted by their counts in acoustic training data, "
          ,file=args.ref_lexicon_edits_handle)
    print("# 1st Col: source of the candidate pron: G(2P) / P(hone-decoding) / R(eference)."
          ,file=args.ref_lexicon_edits_handle)
    print("# 2nd Col: accepted or not in the learned lexicon (Y/N).", file=args.ref_lexicon_edits_handle)
    print("# 3rd Col: soft counts from lattice-alignment (not augmented by prior-counts)."
          ,file=args.ref_lexicon_edits_handle)
    print("# 4th Col: the pronunciation cadidate.", file=args.ref_lexicon_edits_handle)
    
    # words which are to be printed into the edits file.
    words_to_edit = [] 
    num_prons_tot = 0
    for word in learned_lexicon:
        num_prons_tot += len(learned_lexicon[word])
        count = len(stats[word]) # This count could be smaller than the count read from the dict "counts",
        # since in each sub-utterance, multiple occurences (which is rare) of the same word are compressed into one.
        # We use this count here so that in the edit-file, soft counts for each word sum up to one. 
        flags = ['0' for i in range(3)] # "flags" contains three binary indicators, 
        # indicating where this word's pronunciations come from.
        for pron in learned_lexicon[word]:
            if word in pd_lexicon and pron in pd_lexicon[word]:
                flags[0] = '1'
                src[(word, pron)] = 'P'
            elif word in ref_lexicon and pron in ref_lexicon[word]:
                flags[1] = '1'
                src[(word, pron)] = 'R'
            elif word in g2p_lexicon and pron in g2p_lexicon[word]:
                flags[2] = '1'
                src[(word, pron)] = 'G'
        if word in ref_lexicon:
            all_ref_prons_accepted = True
            for pron in ref_lexicon[word]:
                if pron not in learned_lexicon[word]:
                    all_ref_prons_accepted = False
                    break
            if not all_ref_prons_accepted or flags[0] == '1' or flags[2] == '1':
                words_to_edit.append((word, len(stats[word])))
            if count > threshold:
                words[0][flags[0] + flags[1] + flags[2]].add(word)
            else:
                words[1][flags[0] + flags[1] + flags[2]].add(word)
        else:
            if count > threshold: 
                words[2][flags[0] + flags[2]].add(word)
            else:
                words[3][flags[0] + flags[2]].add(word)

    words_to_edit_sorted = sorted(words_to_edit, key=lambda entry: entry[1], reverse=True)
    for word, count in words_to_edit_sorted:
        print("------------",word, "%2.1f" % count, "--------------", file=args.ref_lexicon_edits_handle)
        learned_prons = []
        for pron in learned_lexicon[word]:
            learned_prons.append((src[(word, pron)], 'Y', stats_summed[(word, pron)], pron))
        for pron in ref_lexicon[word]:
            if pron not in learned_lexicon[word]:
                learned_prons.append(('R', 'N', stats_summed[(word, pron)], pron))
        learned_prons_sorted = sorted(learned_prons, key=lambda item: item[2], reverse=True)
        for item in learned_prons_sorted:
            print('{} | {} |  {:.2f} | {}'.format(item[0], item[1], item[2], item[3]), file=args.ref_lexicon_edits_handle)

    num_oovs_with_acoustic_evidence = len(set(learned_lexicon.keys()).difference(set(ref_lexicon.keys())))
    num_oovs = len(set(counts.keys()).difference(set(ref_lexicon.keys())))
    num_ivs = len(learned_lexicon) - num_oovs_with_acoustic_evidence
    print("Average num. prons per word in the learned lexicon is {}".format(float(num_prons_tot)/float(len(learned_lexicon))), file=sys.stderr)
    # print("Here are the words whose reference pron candidates were all declined", words[0]['100'], file=sys.stderr)
    print("-------------------------------------------------Summary------------------------------------------", file=sys.stderr)
    print("We have acoustic evidence for {} out of {} in-vocab (w.r.t the reference lexicon) words from the acoustic training data.".format(num_ivs, len(ref_lexicon)), file=sys.stderr) 
    print("  Among those frequent words whose counts in the training text > ", threshold, ":", file=sys.stderr) 
    num_freq_ivs_from_all_sources = len(words[0]['111']) + len(words[0]['110']) + len(words[0]['011'])
    num_freq_ivs_from_g2p_or_phonetic_decoding = len(words[0]['101']) + len(words[0]['001']) + len(words[0]['100'])
    num_freq_ivs_from_ref = len(words[0]['010'])
    num_infreq_ivs_from_all_sources = len(words[1]['111']) + len(words[1]['110']) + len(words[1]['011'])
    num_infreq_ivs_from_g2p_or_phonetic_decoding = len(words[1]['101']) + len(words[1]['001']) + len(words[1]['100'])
    num_infreq_ivs_from_ref = len(words[1]['010'])
    print('    {} words\' selected prons came from the reference lexicon, G2P/phonetic-decoding.'.format(num_freq_ivs_from_all_sources), file=sys.stderr)
    print('    {} words\' selected prons come from G2P/phonetic-decoding-generated.'.format(num_freq_ivs_from_g2p_or_phonetic_decoding), file=sys.stderr) 
    print('    {} words\' selected prons came from the reference lexicon only.'.format(num_freq_ivs_from_ref), file=sys.stderr) 
    print('  For those words whose counts in the training text <= {}:'.format(threshold), file=sys.stderr) 
    print('    {} words\' selected prons came from the reference lexicon, G2P/phonetic-decoding.'.format(num_infreq_ivs_from_all_sources), file=sys.stderr)
    print('    {} words\' selected prons come from G2P/phonetic-decoding-generated.'.format(num_infreq_ivs_from_g2p_or_phonetic_decoding), file=sys.stderr) 
    print('    {} words\' selected prons came from the reference lexicon only.'.format(num_infreq_ivs_from_ref), file=sys.stderr) 
    print("---------------------------------------------------------------------------------------------------", file=sys.stderr)
    num_freq_oovs_from_both_sources = len(words[2]['11'])
    num_freq_oovs_from_phonetic_decoding = len(words[2]['10'])
    num_freq_oovs_from_g2p = len(words[2]['01'])
    num_infreq_oovs_from_both_sources = len(words[3]['11'])
    num_infreq_oovs_from_phonetic_decoding = len(words[3]['10'])
    num_infreq_oovs_from_g2p = len(words[3]['01'])
    print('We have acoustic evidence for {} out of {} OOV (w.r.t the reference lexicon) words from the acoustic training data.'.format(num_oovs_with_acoustic_evidence, num_oovs), file=sys.stderr)
    print('  Among those words whose counts in the training text > {}:'.format(threshold), file=sys.stderr)
    print('    {} words\' selected prons came from G2P and phonetic-decoding.'.format(num_freq_oovs_from_both_sources), file=sys.stderr)
    print('    {} words\' selected prons came from phonetic decoding only.'.format(num_freq_oovs_from_phonetic_decoding), file=sys.stderr) 
    print('    {} words\' selected prons came from G2P only.'.format(num_freq_oovs_from_g2p), file=sys.stderr) 
    print('  For those words whose counts in the training text <= {}:'.format(threshold), file=sys.stderr) 
    print('    {} words\' selected prons came from G2P and phonetic-decoding.'.format(num_infreq_oovs_from_both_sources), file=sys.stderr)
    print('    {} words\' selected prons came from phonetic decoding only.'.format(num_infreq_oovs_from_phonetic_decoding), file=sys.stderr) 
    print('    {} words\' selected prons came from G2P only.'.format(num_infreq_oovs_from_g2p), file=sys.stderr) 

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
    pd_lexicon =  ReadLexicon(args, args.pd_lexicon_handle, counts)
    stats, stats_summed = ReadArcStats(args.arc_stats_file_handle)
    learned_lexicon =  ReadLexicon(args, args.learned_lexicon_handle, counts)
    
    # Write the learned prons for words out of the ref. vocab into learned_lexicon_oov.
    WriteLearnedLexiconOov(learned_lexicon, ref_lexicon, args.learned_lexicon_oov_handle)
    # Edits will be printed into ref_lexicon_edits, and the summary will be printed into stderr.
    WriteEditsAndSummary(args, learned_lexicon, ref_lexicon, pd_lexicon, g2p_lexicon, counts, stats, stats_summed)

if __name__ == "__main__":
    Main()
