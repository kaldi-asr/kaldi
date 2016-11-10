#!/usr/bin/env python

# Copyright 2016  Johns Hopkins University (Author: Daniel Povey)
# Apache 2.0.

from __future__ import print_function
import sys
import argparse
import math
from collections import defaultdict

# note, this was originally based

parser = argparse.ArgumentParser(description="""
This script converts an ARPA-format language model to FST format
(like the C++ program arpa2fst), but does so while applying bigram
constraints supplied in a separate file.  The resulting language
model will have no unigram state, and there will be no backoff from
the bigram level.
This is useful for phone-level language models in order to keep
graphs small and impose things like linguistic constraints on
allowable phone sequences.
This script writes its output to the stdout.  It is a text-form FST,
suitable for compilation by fstcompile.
""")


parser.add_argument('--disambig-symbol', type = str, default = "#0",
                    help = 'Disambiguation symbol (e.g. #0), '
                    'that is printed on the input side only of backoff '
                    'arcs (output side would be epsilon)')
parser.add_argument('arpa_in', type = str,
                    help = 'The input ARPA file (must not be gzipped)')
parser.add_argument('allowed_bigrams_in', type = str,
                    help = "A file containing the list of allowed bigram pairs.  "
                    "Must include pairs like '<s> foo' and 'foo </s>', as well as "
                    "pairs like 'foo bar'.")
parser.add_argument('--verbose', type = int, default = 0,
                    choices=[0,1,2,3,4,5], help = 'Verbose level')

args = parser.parse_args()

if args.verbose >= 1:
    print(' '.join(sys.argv), file = sys.stderr)


class HistoryState:
    def __init__(self):
        # note: neither backoff_prob nor the floats
        # in word_to_prob are in log space.
        self.backoff_prob = 1.0
        # will be a dict from string to float.  the prob is
        # the actual probability of the word, including any probability
        # mass from backoff (they get added together while writing out
        # the arpa, and these probs are read in from the arpa).
        self.word_to_prob = dict()


class ArpaModel:
    def __init__(self):
        # self.orders is indexed by history-length [i.e. 0 for unigram,
        # 1 for bigram and so on], and is then a dict indexed
        # by tuples of history-words.  E.g. for trigrams, we'd index
        # it as self.orders[2][('a', 'b')].
        # The value-type of the dict is HistoryState.  E.g. to set the
        # probability of the trigram a b -> c to 0.2, we'd do
        # self.orders[2][('a', 'b')].word_to_prob['c'] = 0.2
        self.orders = []

    def Read(self, arpa_in):
        assert len(self.orders) == 0
        log10 = math.log(10.0)
        if arpa_in == "" or arpa_in == "-":
            arpa_in = "/dev/stdin"
        try:
            f = open(arpa_in, "r")
        except:
            sys.exit("{0}: error opening ARPA file {1}".format(
                     sys.argv[0], arpa_in))
        # first read till the \data\ marker.
        while True:
            line = f.readline()
            if line == '':
                sys.exit("{0}: reading {1}, got EOF looking for \\data\\ marker.".format(
                    sys.argv[0], arpa_in))
            if line[0:6] == '\\data\\':
                break
        while True:
            # read, and ignore, the lines like 'ngram 1=1264'...
            line = f.readline()
            if line == '\n' or line == '\r\n':
                break
            if line[0:5] != 'ngram':
                sys.exit("{0}: reading {1}, read something unexpected in header: {2}".format(
                    sys.argv[0], arpa_in, line[:-1]))
            rest=line[5:]
            a = rest.split('=')  # e.g. a = [ '1', '1264] ]
            if len(a) != 2:
                sys.exit("{0}: reading {1}, read something unexpected in header: {2}".format(
                    sys.argv[0], arpa_in, line[:-1]))
            max_order = int(a[0])


        for n in range(max_order):
            # self.orders[n], indexed by history-length (length of the
            # history-vector, == order-1), is a map from history as a tuple
            # of strings, to class HistoryState.
            self.orders.append(defaultdict(lambda: HistoryState()))

        cur_order = 0
        while True:
            line = f.readline()
            if line == '':
                sys.exit("{0}: reading {1}, found EOF while looking for \\end\\ marker.".format(
                    sys.argv[0], arpa_in))
            elif line[0:5] == '\\end\\':
                if len(self.orders) == 0:
                    sys.exit("{0}: reading {1}, read no n-grams.".format(sys.argv[0], arpa_in))
                break
            else:
                cur_order += 1
                expected_line = '\\{0}-grams:'.format(cur_order)
                if not expected_line in line:  # e.g. allow trailing whitespace and newline
                    sys.exit("{0}: reading {1}, expected line {1}, got {2}".format(arpa_in, expected_line, line[:-1]))
                if args.verbose >= 2:
                    print("{0}: reading {1}-grams".format(
                        sys.argv[0], cur_order), file = sys.stderr)

                # now read all the n-grams from this order.
                while True:
                    line = f.readline()
                    # the section of n-grams is terminated by a blank line.
                    if line == '\n' or line == '\r\n':
                        break
                    a = line.split()
                    l = len(a)
                    if l != cur_order + 1 and l != cur_order + 2:
                        sys.exit("{0}: reading {1}: in {2}-grams section, got bad line: {3}".format(
                            sys.argv[0], arpa_in, cur_order, line[:-1]))
                    try:
                        prob = math.exp(float(a[0]) * log10)
                        hist = tuple(a[1:cur_order])  # tuple of strings
                        word = a[cur_order]  # a string
                        backoff_prob = math.exp(float(a[cur_order+1]) * log10) if l == cur_order + 2 else None
                    except Exception as e:
                        sys.exit("{0}: reading {1}: in {2}-grams section, got bad "
                                 "line (exception is: {3}): {4}".format(
                                     sys.argv[0], arpa_in, cur_order,
                                     str(type(e)) + ',' + str(e), line[:-1]))
                    self.orders[cur_order-1][hist].word_to_prob[word] = prob
                    if backoff_prob != None:
                        self.orders[cur_order][hist + (word,)].backoff_prob = backoff_prob

        if args.verbose >= 2:
            print("{0}: read {1}-gram model from {2}".format(
                sys.argv[0], cur_order, arpa_in), file = sys.stderr)
        if cur_order < 2:
            # we'd have to have some if-statements in the code to make this work,
            # and I don't want to have to test it.
            sys.exit("{0}: this script does not work when the ARPA language model "
                     "is unigram.".format(sys.argv[0]))

    # Returns the probability of word 'word' in history-state 'hist'.
    # Dies with error if this word is not predicted at all by the LM (not in vocab).
    # history-state does not exist.
    def GetProb(self, hist, word):
        assert len(hist) < len(self.orders)
        if len(hist) == 0:
            word_to_prob = self.orders[0][()].word_to_prob
            if not word in word_to_prob:
                sys.exit("{0}: no probability in unigram for word {1}".format(
                    sys.argv[0], word))
            return word_to_prob[word]
        else:
            if hist in self.orders[len(hist)]:
                hist_state = self.orders[len(hist)][hist]
                if word in hist_state.word_to_prob:
                    return hist_state.word_to_prob[word]
                else:
                    return hist_state.backoff_prob * self.GetProb(hist[1:], word)
            else:
                return self.GetProb(hist[1:], word)

    # This gets the state corresponding to 'hist' in 'hist_to_state', but backs
    # off for us if there is no such state.
    def GetStateForHist(self, hist_to_state, hist):
        if hist in hist_to_state:
            return hist_to_state[hist]
        else:
            if len(hist) <= 1:
                # this would likely be a code error, but possibly an error
                # in the ARPA file
                sys.exit("{0}: error processing histories: history-state {1} "
                         "does not exist.".format(sys.argv[0], hist))
            return self.GetStateForHist(hist_to_state, hist[1:])


    def GetHistToStateMap(self):
        # This function, called from PrintAsFst, returns (hist_to_state,
        # state_to_hist), which map from history (as a tuple of strings) to
        # integer FST-state and vice versa.

        hist_to_state = dict()
        state_to_hist = []

        # Make sure the initial bigram state comes first (and that
        # we have such a state even if it was completely pruned
        # away in the bigram LM.. which is unlikely of course)
        hist = ('<s>',)
        hist_to_state[hist] = len(state_to_hist)
        state_to_hist.append(hist)

        # create a bigram state for each of the 'real' words...  even if the LM
        # didn't naturally have such bigram states, we'll create them so that we
        # can enforce the bigram constraints supplied in 'bigrams_file' by the
        # user.
        for word in self.orders[0][()].word_to_prob:
            if word != '<s>' and word != '</s>':
                hist = (word,)
                hist_to_state[hist] = len(state_to_hist)
                state_to_hist.append(hist)

        # note: we do not allocate an FST state for the unigram state, because
        # we don't have a unigram state in the output FST, only bigram states; and
        # we don't iterate over bigram histories because we covered them all above;
        # that's why we start 'n' from 2 below instead of from 0.
        for n in range(2, len(self.orders)):
            for hist in self.orders[n].keys():
                # note: hist is a tuple of strings.
                assert not hist in hist_to_state
                hist_to_state[hist] = len(state_to_hist)
                state_to_hist.append(hist)

        return (hist_to_state, state_to_hist)

    # This function prints the estimated language model as an FST.
    # disambig_symbol will be something like '#0' (a symbol introduced
    # to make the result determinizable).
    # bigram_map represent the allowed bigrams (left-word, right-word): it's a map
    # from left-word to a set of right-words (both are strings).
    def PrintAsFst(self, disambig_symbol, bigram_map):
        # History will map from history (as a tuple) to integer FST-state.
        (hist_to_state, state_to_hist) = self.GetHistToStateMap()


        # The following 3 things are just for diagnostics.
        normalization_stats = [ [0, 0.0] for x in range(len(self.orders)) ]
        num_ngrams_allowed = 0
        num_ngrams_disallowed = 0

        for state in range(len(state_to_hist)):
            hist = state_to_hist[state]
            hist_len = len(hist)
            assert hist_len > 0
            if hist_len == 1:  # it's a bigram state...
                context_word = hist[0]
                if not context_word in bigram_map:
                    print("{0}: warning: word {1} appears in ARPA but is not listed "
                          "as a left context in the bigram map".format(
                              sys.argv[0], context_word), file = sys.stderr)
                    continue
                # word list is a list of words that can follow this word.  It must be nonempty.
                word_list = list(bigram_map[context_word])

                normalization_stats[hist_len][0] += 1

                for word in word_list:
                    prob = self.GetProb((context_word,), word)
                    assert prob != 0
                    normalization_stats[hist_len][1] += prob
                    cost = -math.log(prob)
                    if abs(cost) < 0.01 and args.verbose >= 3:
                        print("{0}: warning: very small cost {1} for {2}->{3}".format(
                            sys.argv[0], cost, context_word, word), file=sys.stderr)
                    if word == '</s>':
                        # print the final-prob of this state.
                        print("%d %.3f" % (state, cost))
                    else:
                        next_state = self.GetStateForHist(hist_to_state,
                                                          (context_word, word))
                        print("%d %d %s %s %.3f" %
                              (state, next_state, word, word, cost))
            else:  # it's a higher-order than bigram state.
                assert hist in self.orders[hist_len]
                hist_state = self.orders[hist_len][hist]
                most_recent_word = hist[-1]

                normalization_stats[hist_len][0] += 1
                normalization_stats[hist_len][1] += \
                  sum([ self.GetProb(hist, word) for word in bigram_map[most_recent_word]])

                for word, prob in hist_state.word_to_prob.items():
                    cost = -math.log(prob)
                    if word in bigram_map[most_recent_word]:
                        num_ngrams_allowed += 1
                    else:
                        num_ngrams_disallowed += 1
                        continue
                    if word == '</s>':
                        # print the final-prob of this state.
                        print("%d %.3f" % (state, cost))
                    else:
                        next_state = self.GetStateForHist(hist_to_state,
                                                          (hist) + (word,))
                        print("%d %d %s %s %.3f" %
                              (state, next_state, word, word, cost))
                # Now deal with the backoff probability of this state (back off
                # to the lower-order state).
                assert hist in self.orders[hist_len]
                backoff_prob = self.orders[hist_len][hist].backoff_prob
                assert backoff_prob != 0.0
                cost = -math.log(backoff_prob)
                backoff_hist = hist[1:]
                backoff_state = self.GetStateForHist(hist_to_state, backoff_hist)
                # note: we only print the disambig symbol on the input side.
                if args.verbose >= 3 and abs(cost) < 0.001:
                    print("{0}: very low backoff cost {1} for history {2}, state = {3}".format(
                        sys.argv[0], cost, str(hist), state), file = sys.stderr)

                # For hist-states that completely back off (they have no words coming out of them),
                # there is no need to disambiguate, we can print an epsilon that will later be removed.
                this_disambig_symbol = disambig_symbol if len(hist_state.word_to_prob) != 0 else '<eps>'
                print("%d %d %s <eps> %.3f" %
                      (state, backoff_state, this_disambig_symbol, cost))
        if args.verbose >= 1:
            for hist_len in range(1, len(self.orders)):
                num_states = normalization_stats[hist_len][0]
                avg_prob_sum = normalization_stats[hist_len][1] / num_states if num_states > 0 else 0.0
                print("{0}: for {1}-gram states, over {2} states the average sum of "
                      "probs was {3} (would be 1.0 if properly normalized).".format(
                          sys.argv[0], hist_len + 1, num_states, avg_prob_sum),
                      file = sys.stderr)
            if num_ngrams_disallowed != 0:
                print("{0}: for explicit n-grams higher than bigram from the ARPA model, {0} "
                      "were allowed by the bigram constraints and {1} were disallowed (we "
                      "normally expect all or almost all of them to be allowed).".format(
                          num_ngrams_allowed, num_ngrams_disallowed), file = sys.stderr)



# returns a map which is a dict [indexed by left-hand word] of sets [containing
# the right-hand word].
def ReadBigramMap(bigrams_file):
    ans = defaultdict(lambda: set())

    have_one_bos = False
    have_one_eos = False
    have_one_regular = False

    try:
        f = open(bigrams_file, "r")
    except:
        sys.exit("utils/lang/internal/arpa2fst_constrained.py: error opening "
                 "bigrams file " + bigrams_file)
    while True:
        line = f.readline()
        if line == '':
            break
        a = line.split()
        if len(a) != 2:
            sys.exit("utils/lang/internal/arpa2fst_constrained.py: bad line in "
                     "bigrams file {0} (expect 2 fields): {1}".format(
                         bigrams_file, line[:-1]))
        [word1, word2] = a
        if word1 in ans and word2 in ans[word1]:
            sys.exit("{0}: bigrams file contained duplicate entry: {1} {2}".format(
                sys.argv[0], word1, word2), file = sys.stderr)
        if word2 == '<s>' or word1 == '</s>':
            sys.exit("{0}: bad sequence of BOS/EOS symbols: {1} {2}".format(
                sys.argv[0], word1, word2))
        if word1 == '<s>':
            have_one_bos = True
        elif word2 == '</s>':
            have_one_eos = True
        else:
            have_one_regular = True
        ans[word1].add(word2)
    # check for at least one pair with BOS
    if len(ans) == 0:
        sys.exit("{0}: no data found in bigrams file {1}".format(
            sys.argv[0], bigrams_file))
    elif not (have_one_bos and have_one_eos and have_one_regular):
        sys.exit("{0}: the bigrams file {1} does not look right "
                 "(make sure BOS and EOS symbols are there)".format(
            sys.argv[0], bigrams_file))
    return ans

arpa_model = ArpaModel()
arpa_model.Read(args.arpa_in)
bigrams_map = ReadBigramMap(args.allowed_bigrams_in)
if len(args.disambig_symbol.split()) != 1:
    sys.exit("{0}: invalid option --disambig-symbol={1}".format(
        sys.argv[0], args.disambig_symbol))
arpa_model.PrintAsFst(args.disambig_symbol, bigrams_map)
