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
        description = "Use a greedy framework to select pronunciation candidates"
        "from three sources: a reference lexicon, G2P lexicon and phonetic-decoding"
        "(PD) lexicon. Basically, this script implements the Alg. 1 in the paper:"
        "Acoustic data-driven lexicon learning based on a greedy pronunciation "
        "selection framework, by X. Zhang, V. Mahonar, D. Povey and S. Khudanpur,"
        "Interspeech 2017. The inputs are an arc-stats file, containing "
        "acoustic evidence (tau_{uwb} in the paper) and three source lexicons "
        "(phonetic-decoding(PD)/G2P/ref). The outputs is the learned lexicon for"
        "all words in the arc_stats (acoustic evidence) file.",
        epilog = "See steps/dict/learn_lexicon_greedy.sh for example.")
    parser.add_argument("--alpha", type = str, default = "0,0,0",
                        help = "Scaling factors for the likelihood reduction threshold."
                        "of three pronunciaiton candidate sources: phonetic-decoding (PD),"
                        "G2P and reference. The valid range of each dimension is [0, 1], and"
                        "a large value means we prune pronunciations from this source more"
                        "aggressively. Setting a dimension to zero means we never want to remove"
                        "pronunciaiton from that source. See Section 4.3 in the paper for details.")
    parser.add_argument("--beta", type = str, default = "0,0,0",
                        help = "smoothing factors for the likelihood reduction term."
                        "of three pronunciaiton candidate sources: phonetic-decoding (PD),"
                        "G2P and reference. The valid range of each dimension is [0, 100], and"
                        "a large value means we prune pronunciations from this source more"
                        "aggressively. See Section 4.3 in the paper for details.")
    parser.add_argument("--delta", type = float, default = 0.000000001,
                        help = "Floor value of the pronunciation posterior statistics."
                        "The valid range is (0, 0.01),"
                        "See Section 3 in the paper for details.")
    parser.add_argument("silence_phones_file", metavar = "<silphone-file>", type = str,
                        help = "File containing a list of silence phones.")
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
    parser.add_argument("pd_lexicon", metavar = "<phonetic-decoding-lexicon>", type = str,
                        help = "Candidate ronouciations from phonetic decoding results."
                        "Each line must be <word> <phones>")
    parser.add_argument("learned_lexicon", metavar = "<learned-lexicon>", type = str,
                        help = "Learned lexicon.")


    print (' '.join(sys.argv), file=sys.stderr)

    args = parser.parse_args()
    args = CheckArgs(args)

    return args

def CheckArgs(args):
    args.silence_phones_file_handle = open(args.silence_phones_file)
    if args.arc_stats_file == "-":
        args.arc_stats_file_handle = sys.stdin
    else:
        args.arc_stats_file_handle = open(args.arc_stats_file)
    args.word_counts_file_handle = open(args.word_counts_file)
    args.ref_lexicon_handle = open(args.ref_lexicon)
    args.g2p_lexicon_handle = open(args.g2p_lexicon)
    args.pd_lexicon_handle = open(args.pd_lexicon)
    args.learned_lexicon_handle = open(args.learned_lexicon, "w")
    
    alpha = args.alpha.strip().split(',')
    if len(alpha) is not 3:
        raise Exception('Invalid alpha ', args.alpha)
    for i in range(0,3):
        if float(alpha[i]) < 0 or float(alpha[i]) > 1:
            raise Exception('alaph ', alpha[i], 
                            ' is invalid, it must be within [0, 1].')
        if float(alpha[i]) == 0:
            alpha[i] = -1e-3
        # The absolute likelihood loss (search for loss_abs) is supposed to be positive.
        # But it could be negative near zero because of numerical precision limit.
        # In this case, even if alpha is set to be zero, which means we never want to
        # remove pronunciation from that source, the quality score (search for q_b)
        # could still be negative, which means this pron could be potentially removed.
        # To prevent this, we set alpha as a negative value near zero to ensure
        # q_b is always positive.

    args.alpha = [float(alpha[0]), float(alpha[1]), float(alpha[2])]
    print("[alpha_{pd}, alpha_{g2p}, alpha_{ref}] is: ", args.alpha)
    exit
    beta = args.beta.strip().split(',')
    if len(beta) is not 3:
        raise Exception('Invalid beta ', args.beta)
    for i in range(0,3):
        if float(beta[i]) < 0 or float(beta[i]) > 100:
            raise Exception('beta ', beta[i], 
                            ' is invalid, it must be within [0, 100].')
    args.beta = [float(beta[0]), float(beta[1]), float(beta[2])]
    print("[beta_{pd}, beta_{g2p}, beta_{ref}] is: ", args.beta)

    if args.delta <= 0 or args.delta > 0.1:
        raise Exception('delta ', args.delta, ' is invalid, it must be within'
                        '(0, 0.01).')
    print("delta is: ", args.delta)

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

def FilterPhoneticDecodingLexicon(args, pd_lexicon):
    # We want to remove all candidates which contain silence phones
    silphones = set()
    for line in args.silence_phones_file_handle:
        silphones.add(line.strip())
    rejected_candidates = set()
    for word, prons in pd_lexicon.iteritems():
        for pron in prons:
            for phone in pron.split():
                if phone in silphones:
                   rejected_candidates.add((word, pron))
                   break
    for word, pron in rejected_candidates:
        pd_lexicon[word].remove(pron)
    return pd_lexicon

# One iteration of Expectation-Maximization computation (Eq. 3-4 in the paper).
def OneEMIter(args, word, stats, prons, pron_probs, debug=False):
    prob_acc = [0.0 for i in range(len(prons[word]))]
    s = sum(pron_probs)
    for i in range(len(pron_probs)):
        pron_probs[i] = pron_probs[i] / s
    log_like = 0.0
    for (utt, start_frame) in stats[word]:
        prob = []
        soft_counts = []
        for i in range(len(prons[word])):
            phones = prons[word][i]
            soft_count = stats[word][(utt, start_frame)].get(phones, 0)
            if soft_count < args.delta: 
                soft_count = args.delta
            soft_counts.append(soft_count)
        prob = [i[0] * i[1] for i in zip(soft_counts, pron_probs)]
        for i in range(len(prons[word])):
            prob_acc[i] += prob[i] / sum(prob)
        log_like += math.log(sum(prob))
    pron_probs = [1.0 / float(len(stats[word])) * p for p in prob_acc]
    log_like = 1.0 / float(len(stats[word])) * log_like
    if debug:
        print("Log_like of the word: ", log_like, "pron probs: ", pron_probs)
    return pron_probs, log_like

def SelectPronsGreedy(args, stats, counts, ref_lexicon, g2p_lexicon, pd_lexicon, dianostic_info=False):
    prons = defaultdict(list) # Put all possible prons from three source lexicons into this dictionary
    src = {} # Source of each (word, pron) pair: 'P' = phonetic-decoding, 'G' = G2P, 'R' = reference
    learned_lexicon = defaultdict(set) # Put all selected prons in this dictionary
    for lexicon in ref_lexicon, g2p_lexicon, pd_lexicon:
        for word in lexicon:
            for pron in lexicon[word]:
                prons[word].append(pron)
    for word in prons:
        for pron in prons[word]:
            if word in pd_lexicon and pron in pd_lexicon[word]:
                src[(word, pron)] = 'P'
            if word in g2p_lexicon and pron in g2p_lexicon[word]:
                src[(word, pron)] = 'G'
            if word in ref_lexicon and pron in ref_lexicon[word]:
                src[(word, pron)] = 'R'
   
    for word in prons:
        if word not in stats:
            continue
        n = len(prons[word])
        pron_probs = [1/float(n) for i in range(n)]
        if dianostic_info:
            print("pronunciations of word '{}': {}".format(word, prons[word]))
        active_indexes = set(range(len(prons[word])))
       
        deleted_prons = [] # indexes of prons to be deleted
        soft_counts_normalized = []
        while len(active_indexes) > 1:
            log_like = 1.0
            log_like_last = -1.0
            num_iters = 0
            while abs(log_like - log_like_last) > 1e-7:
                num_iters += 1
                log_like_last = log_like
                pron_probs, log_like = OneEMIter(args, word, stats, prons, pron_probs, False)
                if log_like_last == 1.0 and len(soft_counts_normalized) == 0: # the first iteration
                    soft_counts_normalized = pron_probs
                    if dianostic_info: 
                        print("Avg.(over all egs) soft counts: {}".format(soft_counts_normalized))
            if dianostic_info:
                print("\n Log_like after {} iters of EM: {}, estimated pron_probs: {} \n".format(
                        num_iters, log_like, pron_probs))
            candidates_to_delete = []
            
            for i in active_indexes:
                pron_probs_mod = [p for p in pron_probs]
                pron_probs_mod[i] = 0.0
                for j in range(len(pron_probs_mod)):
                    if j in active_indexes and j != i:
                        pron_probs_mod[j] += 0.01
                pron_probs_mod = [s / sum(pron_probs_mod) for s in pron_probs_mod]
                log_like2 = 1.0
                log_like2_last = -1.0
                num_iters2 = 0
                # Running EM until convengence
                while abs(log_like2 - log_like2_last) > 0.001 :
                    num_iters2 += 1
                    log_like2_last = log_like2
                    pron_probs_mod, log_like2 = OneEMIter(args, word, stats,
                                                          prons, pron_probs_mod, False)
                
                loss_abs = log_like - log_like2 # absolute likelihood loss before normalization
                # (supposed to be positive, but could be negative near zero because of numerical precision limit).
                log_delta = math.log(args.delta)
                thr = -log_delta
                loss = loss_abs
                source = src[(word, prons[word][i])]
                if dianostic_info:
                    print("\n set the pron_prob of '{}' whose source is {}, to zero results in {}"
                    " loss in avg. log-likelihood; Num. iters until converging:{}. ".format(
                      prons[word][i], source, loss, num_iters2))
                # Compute quality score q_b = loss_abs * / (M_w + beta_s(b)) + alpha_s(b) * log_delta
                # See Sec. 4.3 and Alg. 1 in the paper.
                if source == 'P':
                   thr *= args.alpha[0]
                   loss *= float(len(stats[word])) / (float(len(stats[word])) + args.beta[0])
                if source == 'G':
                   thr *= args.alpha[1]
                   loss *= float(len(stats[word])) / (float(len(stats[word])) + args.beta[1])
                if source == 'R':
                   thr *= args.alpha[2]
                   loss *= float(len(stats[word])) / (float(len(stats[word])) + args.beta[2])
                if loss - thr < 0: # loss - thr here is just q_b
                   if dianostic_info:
                       print("Smoothed log-like loss {} is smaller than threshold {} so that the quality"
                             "score {} is negative, adding the pron to the list of candidates to delete"
                             ". ".format(loss, thr, loss-thr))
                   candidates_to_delete.append((loss-thr, i))
            if len(candidates_to_delete) == 0:
                break
            candidates_to_delete_sorted = sorted(candidates_to_delete, 
                                                 key=lambda candidates_to_delete: candidates_to_delete[0])

            deleted_candidate = candidates_to_delete_sorted[0]
            active_indexes.remove(deleted_candidate[1])
            pron_probs[deleted_candidate[1]] = 0.0
            for i in range(len(pron_probs)):
                if i in active_indexes:
                    pron_probs[i] += 0.01
            pron_probs = [s / sum(pron_probs) for s in pron_probs]
            source = src[(word, prons[word][deleted_candidate[1]])]
            pron = prons[word][deleted_candidate[1]]
            soft_count = soft_counts_normalized[deleted_candidate[1]]
            quality_score = deleted_candidate[0]
            # This part of diagnostic info provides hints to the user on how to adjust the parameters.
            if dianostic_info:
                print("removed pron {}, from source {} with quality score {:.5f}".format(
                        pron, source, quality_score)) 
                if (source == 'P' and soft_count > 0.7 and len(stats[word]) > 5):
                    print("WARNING: alpha_{pd} or beta_{pd} may be too large!"
                          "    For the word '{}' whose count is {}, the candidate "
                          "    pronunciation from phonetic decoding '{}' with normalized "
                          "    soft count {} (out of 1) is rejected. It shouldn't have been"
                          "    rejected if alpha_{pd} is smaller than {}".format(
                            word, len(stats[word]), pron, soft_count, -loss / log_delta, 
                            -args.alpha[0] * len(stats[word]) + (objf_change + args.beta[0])),
                            file=sys.stderr)
                    if loss_abs > thr:
                        print("    or beta_{pd} is smaller than {}".format(
                                (loss_abs / thr - 1) * len(stats[word])), file=sys.stderr)
                if (source == 'G' and soft_count > 0.7 and len(stats[word]) > 5):
                    print("WARNING: alpha_{g2p} or beta_{g2p} may be too large!"
                          "    For the word '{}' whose count is {}, the candidate "
                          "    pronunciation from G2P '{}' with normalized "
                          "    soft count {} (out of 1) is rejected. It shouldn't have been"
                          "    rejected if alpha_{g2p} is smaller than {} ".format(
                            word, len(stats[word]), pron, soft_count, -loss / log_delta, 
                            -args.alpha[1] * len(stats[word]) + (objf_change + args.beta[1])),
                          file=sys.stderr)
                    if loss_abs > thr:
                        print("    or beta_{g2p} is smaller than {}.".format((
                                loss_abs / thr - 1) * len(stats[word])), file=sys.stderr)
            deleted_prons.append(deleted_candidate[1])
        for i in range(len(prons[word])):
            if i not in deleted_prons:
                learned_lexicon[word].add(prons[word][i])

    return learned_lexicon

def WriteLearnedLexicon(learned_lexicon, file_handle):
    for word, prons in learned_lexicon.iteritems():
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
    pd_lexicon = FilterPhoneticDecodingLexicon(args, pd_lexicon)
                  
    # Select prons to construct the learned lexicon.
    learned_lexicon = SelectPronsGreedy(args, stats, counts, ref_lexicon, g2p_lexicon, pd_lexicon)
    
    # Write the learned prons for words out of the ref. vocab into learned_lexicon_oov.
    WriteLearnedLexicon(learned_lexicon, args.learned_lexicon_handle)

if __name__ == "__main__":
    Main()
