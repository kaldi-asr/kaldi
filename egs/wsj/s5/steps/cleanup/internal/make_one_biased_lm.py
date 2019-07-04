#!/usr/bin/env python3

# Copyright 2016  Johns Hopkins University (Author: Daniel Povey)
# Apache 2.0.

from __future__ import print_function
from __future__ import division
import sys
import argparse
import math
from collections import defaultdict

import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer,encoding="utf8")
sys.stderr = io.TextIOWrapper(sys.stderr.buffer,encoding="utf8")
sys.stdin = io.TextIOWrapper(sys.stdin.buffer,encoding="utf8")

parser = argparse.ArgumentParser(description="""
This script creates a biased language model suitable for alignment and
data-cleanup purposes.   It reads (possibly multiple) lines of integerized text
from the input and writes a text-form FST of a backoff language model to
the standard output, to be piped into fstcompile.""")

parser.add_argument("--word-disambig-symbol", type = int, required = True,
                    help = "Integer corresponding to the disambiguation "
                    "symbol (normally #0) for backoff arcs")
parser.add_argument("--ngram-order", type = int, default = 4,
                    choices = [2,3,4,5,6,7],
                    help = "Maximum order of n-gram to use (but see also "
                    "--min-lm-state-count; the effective order may be less.")
parser.add_argument("--min-lm-state-count", type = int, default = 10,
                    help = "Minimum count below which we will completely "
                    "discount an LM-state (if it is of order > 2, i.e. "
                    "history-length > 1).")
parser.add_argument("--top-words", type = str,
                    help = "File containing frequent words and probabilities to be added into "
                    "the language model, with lines in the format '<integer-id-of-word> <prob>'. "
                    "These probabilities will be added to the probabilities in the unigram "
                    "backoff state and then renormalized; this option allows you to introduce "
                    "common words to the LM with specified probabilities.")
parser.add_argument("--discounting-constant", type = float, default = 0.3,
                    help = "Discounting constant D for standard (unmodified) Kneser-Ney; "
                    "must be strictly between 0 and 1.  A value closer to 0 will give "
                    "you a more-strongly-biased LM.")
parser.add_argument("--verbose", type = int, default = 0,
                    choices=[0,1,2,3,4,5], help = "Verbose level")

args = parser.parse_args()

if args.verbose >= 1:
    print(' '.join(sys.argv), file = sys.stderr)




class NgramCounts(object):
    ## A note on data-structure.
    ## Firstly, all words are represented as integers.
    ## We store n-gram counts as an array, indexed by (history-length == n-gram order minus one)
    ## (note: python calls arrays "lists")  of dicts from histories to counts, where
    ## histories are arrays of integers and "counts" are dicts from integer to float.
    ## For instance, when accumulating the 4-gram count for the '8' in the sequence '5 6 7 8',
    ## we'd do as follows:
    ##  self.counts[3][[5,6,7]][8] += 1.0
    ## where the [3] indexes an array, the [[5,6,7]] indexes a dict, and
    ## the [8] indexes a dict.
    def __init__(self, ngram_order):
        self.ngram_order = ngram_order
        # Integerized counts will never contain negative numbers, so
        # inside this program, we use -3 and -2 for the BOS and EOS symbols
        # respectively.
        # Note: it's actually important that the bos-symbol is the most negative;
        # it helps ensure that we print the state with left-context <s> first
        # when we print the FST, and this means that the start-state will have
        # the correct value.
        self.bos_symbol = -3
        self.eos_symbol = -2
        # backoff_symbol is kind of a pseudo-word, it's used in keeping track of
        # the backoff counts in each state.
        self.backoff_symbol = -1
        self.counts = []
        for n in range(ngram_order):
            # The 'lambda: defaultdict(float)' is an anonymous function taking
            # no arguments that returns a new defaultdict(float).
            # If we index self.counts[n][history] for a history-length n < ngram_order
            # and a previously unseen history, it will create a new defaultdict
            # that defaults to 0.0 [since the function float() will return 0.0].
            # This means that we can index self.counts without worrying about
            # undefined values.
            self.counts.append(defaultdict(lambda: defaultdict(float)))

    # adds a raw count (called while processing input data).
    # Suppose we see the sequence '6 7 8 9' and ngram_order=4, 'history'
    # would be (6,7,8) and 'predicted_word' would be 9; 'count' would be
    # 1.0.
    def AddCount(self, history, predicted_word, count):
        self.counts[len(history)][history][predicted_word] += count

    # 'line' is a string containing a sequence of integer word-ids.
    # This function adds the un-smoothed counts from this line of text.
    def AddRawCountsFromLine(self, line):
        try:
            words = [self.bos_symbol] + [ int(x) for x in line.split() ] + [self.eos_symbol]
        except:
            sys.exit("make_one_biased_lm.py: bad input line {0} (expected a sequence "
                     "of integers)".format(line))

        for n in range(1, len(words)):
            predicted_word = words[n]
            history_start = max(0, n + 1 - self.ngram_order)
            history = tuple(words[history_start:n])
            self.AddCount(history, predicted_word, 1.0)

    def AddRawCountsFromStandardInput(self):
        lines_processed = 0
        while True:
            line = sys.stdin.readline()
            if line == '':
                break
            self.AddRawCountsFromLine(line)
            lines_processed += 1
        if lines_processed == 0 or args.verbose > 0:
            print("make_one_biased_lm.py: processed {0} lines of input".format(
                    lines_processed), file = sys.stderr)


    # This function returns a dict from history (as a tuple of integers of
    # length > 1, ignoring lower-order histories), to the total count of this
    # history state plus all history-states which back off to this history state.
    # It's used inside CompletelyDiscountLowCountStates().
    def GetHistToTotalCount(self):
        ans = defaultdict(float)
        for n in range(2, self.ngram_order):
            for hist, word_to_count in self.counts[n].items():
                total_count = sum(word_to_count.values())
                while len(hist) >= 2:
                    ans[hist] += total_count
                    hist = hist[1:]
        return ans


    # This function will completely discount the counts in any LM-states of
    # order > 2 (i.e. history-length > 1) that have total count below
    # 'min_count'; when computing the total counts, we include higher-order
    # LM-states that would back off to 'this' lm-state, in the total.
    def CompletelyDiscountLowCountStates(self, min_count):
        hist_to_total_count = self.GetHistToTotalCount()
        for n in reversed(list(range(2, self.ngram_order))):
            this_order_counts = self.counts[n]
            to_delete = []
            for hist in this_order_counts.keys():
                if hist_to_total_count[hist] < min_count:
                    # we need to completely back off this count.
                    word_to_count = this_order_counts[hist]
                    # mark this key for deleting
                    to_delete.append(hist)
                    backoff_hist = hist[1:]  # this will be a tuple not a list.
                    for word, count in word_to_count.items():
                        self.AddCount(backoff_hist, word, count)
            for hist in to_delete:
                del this_order_counts[hist]

    # This backs off the counts according to Kneser-Ney (unmodified,
    # with interpolation).
    def ApplyBackoff(self, D):
        assert D > 0.0 and D < 1.0
        for n in reversed(list(range(1, self.ngram_order))):
            this_order_counts = self.counts[n]
            for hist, word_to_count in this_order_counts.items():
                backoff_hist = hist[1:]
                backoff_word_to_count = self.counts[n-1][backoff_hist]
                this_discount_total = 0.0
                for word in word_to_count:
                    assert word_to_count[word] >= 1.0
                    word_to_count[word] -= D
                    this_discount_total += D
                    # Interpret the following line as incrementing the
                    # count-of-counts for the next-lower order.
                    backoff_word_to_count[word] += 1.0
                word_to_count[self.backoff_symbol] += this_discount_total


    # This function prints out to stderr the n-gram counts stored in this
    # object; it's used for debugging.
    def Print(self, info_string):
        print(info_string, file=sys.stderr)
        # these are useful for debug.
        total = 0.0
        total_excluding_backoff = 0.0
        for this_order_counts in self.counts:
            for hist, word_to_count in this_order_counts.items():
                this_total_count = sum(word_to_count.values())
                print('{0}: total={1} '.format(hist, this_total_count),
                      end='', file=sys.stderr)
                print(' '.join(['{0} -> {1} '.format(word, count)
                                for word, count in word_to_count.items() ]),
                      file = sys.stderr)
                total += this_total_count
                total_excluding_backoff += this_total_count
                if self.backoff_symbol in word_to_count:
                    total_excluding_backoff -= word_to_count[self.backoff_symbol]
        print('total count = {0}, excluding discount = {1}'.format(
                total, total_excluding_backoff), file = sys.stderr)

    def AddTopWords(self, top_words_file):
        empty_history = ()
        word_to_count = self.counts[0][empty_history]
        total = sum(word_to_count.values())
        try:
            f = open(top_words_file, mode='r', encoding='utf-8')
        except:
            sys.exit("make_one_biased_lm.py: error opening top-words file: "
                     "--top-words=" + top_words_file)
        while True:
            line = f.readline()
            if line == '':
                break
            try:
                [ word_index, prob ] = line.split()
                word_index = int(word_index)
                prob = float(prob)
                assert word_index > 0 and prob > 0.0
                word_to_count[word_index] += prob * total
            except Exception as e:
                sys.exit("make_one_biased_lm.py: could not make sense of the "
                         "line '{0}' in op-words file: {1} ".format(line, str(e)))
        f.close()


    def GetTotalCountMap(self):
        # This function, called from PrintAsFst, returns a map from
        # history to the total-count for that state.
        total_count_map = dict()
        for n in range(0, self.ngram_order):
            for hist, word_to_count in self.counts[n].items():
                total_count_map[hist] = sum(word_to_count.values())
        return total_count_map

    def GetHistToStateMap(self):
        # This function, called from PrintAsFst, returns a map from
        # history to integer FST-state.
        hist_to_state = dict()
        fst_state_counter = 0
        for n in range(0, self.ngram_order):
            for hist in self.counts[n].keys():
                hist_to_state[hist] = fst_state_counter
                fst_state_counter += 1
        return hist_to_state

    def GetProb(self, hist, word, total_count_map):
        total_count = total_count_map[hist]
        word_to_count = self.counts[len(hist)][hist]
        prob = float(word_to_count[word]) / total_count
        if len(hist) > 0 and word != self.backoff_symbol:
            prob_in_backoff = self.GetProb(hist[1:], word, total_count_map)
            backoff_prob = float(word_to_count[self.backoff_symbol]) / total_count
            prob += backoff_prob * prob_in_backoff
        return prob

    # This function prints the estimated language model as an FST.
    def PrintAsFst(self, word_disambig_symbol):
        # n is the history-length (== order + 1).  We iterate over the
        # history-length in the order 1, 0, 2, 3, and then iterate over the
        # histories of each order in sorted order.  Putting order 1 first
        # and sorting on the histories
        # ensures that the bigram state with <s> as the left context comes first.
        # (note: self.bos_symbol is the most negative symbol)

        # History will map from history (as a tuple) to integer FST-state.
        hist_to_state = self.GetHistToStateMap()
        total_count_map = self.GetTotalCountMap()

        for n in [ 1, 0 ] + list(range(2, self.ngram_order)):
            this_order_counts = self.counts[n]
            # For order 1, make sure the keys are sorted.
            keys = this_order_counts.keys() if n != 1 else sorted(this_order_counts.keys())
            for hist in keys:
                word_to_count = this_order_counts[hist]
                this_fst_state = hist_to_state[hist]

                for word in word_to_count.keys():
                    # work out this_cost.  Costs in OpenFst are negative logs.
                    this_cost = -math.log(self.GetProb(hist, word, total_count_map))

                    if word > 0: # a real word.
                        next_hist = hist + (word,)  # appending tuples
                        while not next_hist in hist_to_state:
                            next_hist = next_hist[1:]
                        next_fst_state = hist_to_state[next_hist]
                        print(this_fst_state, next_fst_state, word, word,
                              this_cost)
                    elif word == self.eos_symbol:
                        # print final-prob for this state.
                        print(this_fst_state, this_cost)
                    else:
                        assert word == self.backoff_symbol
                        backoff_fst_state = hist_to_state[hist[1:len(hist)]]
                        print(this_fst_state, backoff_fst_state,
                              word_disambig_symbol, 0, this_cost)


ngram_counts = NgramCounts(args.ngram_order)
ngram_counts.AddRawCountsFromStandardInput()

if args.verbose >= 3:
    ngram_counts.Print("Raw counts:")
ngram_counts.CompletelyDiscountLowCountStates(args.min_lm_state_count)
if args.verbose >= 3:
    ngram_counts.Print("Counts after discounting low-count states:")
ngram_counts.ApplyBackoff(args.discounting_constant)
if args.verbose >= 3:
    ngram_counts.Print("Counts after applying Kneser-Ney discounting:")
if args.top_words != None:
    ngram_counts.AddTopWords(args.top_words)
    if args.verbose >= 3:
        ngram_counts.Print("Counts after applying top-n-words")
ngram_counts.PrintAsFst(args.word_disambig_symbol)


# test comand:
# (echo 6 7 8 4; echo 7 8 9; echo 7 8) | ./make_one_biased_lm.py --word-disambig-symbol=1000 --min-lm-state-count=2 --verbose=3 --top-words=<(echo 1 0.5; echo 2 0.25)
