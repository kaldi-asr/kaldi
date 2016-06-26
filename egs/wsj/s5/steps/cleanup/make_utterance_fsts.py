#!/usr/bin/env python

from __future__ import print_function
import sys
import argparse
import math

verbose_level = 0

class Ngrams:
    def __init__(self, oov_id, backoff_arc_id,
                 ngram_order, interpolation_weights,
                 top_word_probs = None,
                 top_words_interpolation_weight = 1.0):
        self.oov_id = oov_id
        self.backoff_arc_id = backoff_arc_id

        self.stats = dict()     # Counts for each ngram
        self.ngrams = dict()    # List of ngrams of different order
        self.histories = dict() # Counts for each ngram history

        self.ngram_order = ngram_order

        if ngram_order not in [1,2]:
            raise ValueError("ngram-order must be 1 or 2")

        self.top_word_probs = (None if top_word_probs is None or len(top_word_probs) == 0
                                    else top_word_probs)

        self.top_words_interpolation_weight = (top_words_interpolation_weight
                                               if self.top_word_probs is not None
                                               else 0)

        if len(interpolation_weights) != (ngram_order+1):
            raise ValueError("Mismatch between interpolation_weights size and (ngram order + 1); {0} vs {1}".format(len(interpolation_weights), ngram_order + 1))

        total_weight = self.top_words_interpolation_weight + sum([x for x in interpolation_weights])
        self.interpolation_weights = [ x / total_weight for x in interpolation_weights ]
        self.top_words_interpolation_weight /= total_weight

    def ResetStats(self):
        self.stats.clear()
        self.ngrams.clear()
        self.histories.clear()

    def AddStats(self, tup, count = 1.0):
        order = self.ngram_order

        # Accumulate stats for all order less or equal to self.ngram_order
        for o in range(1, order+1):
            i = len(tup) - o
            ngram = tup[i:len(tup)]

            assert(type(ngram) == tuple)
            assert(len(ngram) == o)

            if ngram not in self.stats:
                self.stats[ngram] = count
                self.ngrams.setdefault(o, []).append(ngram)
            else:
                self.stats[ngram] += count

            if o > 1:
                hist = ngram[0:-1]
                self.histories[hist] = self.histories.setdefault(hist, 0) + count

    def GetProbs(self, order):
        assert(order >= 0)

        if (order > self.ngram_order):
            raise Exception("ngram_order must be less than the ngram_order of the stats stored; {0} vs {1}".format(order, self.ngram_order))
        probs = dict()

        num_words = len(self.ngrams[1])
        if order == 0:
            for n in self.ngrams[1]:
                probs[n] = 1.0 / num_words
            if verbose_level > 2:
                print ("{0}-gram probs: {1}".format(order, probs), file = sys.stderr)
            return probs

        if order == 1:
            for n in self.ngrams[1]:
                probs[n] = float(self.stats[n]) / num_words
            if verbose_level > 2:
                print ("{0}-gram probs: {1}".format(order, probs), file = sys.stderr)
            return probs

        for n in self.ngrams[order]:
            h = n[0:(order-1)]
            probs[n] = float(self.stats[n]) / self.histories[h]

        if verbose_level > 2:
            print ("{0}-gram probs: {1}".format(order, probs), file = sys.stderr)

        return probs

    def GetFst(self, order, probs):
        fst = []
        history2state = dict()
        num_states = 3

        start_state = 0
        unigram_state = 1
        final_state = 2

        assert(type(probs) == list)
        assert(len(probs) > order)

        if order > 2:
            raise NotImplementedError("ngram-order > 2 not supported")
            history2state[tuple(["<s>"] * order - 1)] = start_state
        elif order == 2:
            history2state[("<s>",)] = start_state
            for tup, prob in probs[order].iteritems():
                h = tup[0:-1]
                if h in history2state:
                    s = history2state[h]
                else:
                    # Create a new state for the history h
                    history2state[h] = num_states
                    s = num_states
                    num_states += 1

                word = tup[-1]      # observed word
                if word == "</s>":
                    # end-of-sentence; go to the final_state
                    word = 0
                    fst.append((s, final_state, word, word,
                                -math.log(prob * self.interpolation_weights[order])))
                else:
                    to_h = tup[1:]      # destination history

                    if to_h in history2state:
                        e = history2state[to_h]
                    else:
                        # Create a new state for the history to_h
                        history2state[to_h] = num_states
                        e = num_states
                        num_states += 1

                    fst.append((s, e, word, word,
                                -math.log(prob * self.interpolation_weights[order])))

        if order <= 2:
            # Assume ngram order <= 2
            unigram_probs = probs[1]
            assert(type(unigram_probs) == dict)

            if self.top_word_probs is not None:
                for word, prob in self.top_word_probs.iteritems():
                    # Interpolate the top word unigram probabilities with the
                    # utterance-specific unigram
                    unigram_probs[(word,)] = (unigram_probs.setdefault((word,), 0) * self.interpolation_weights[1]
                                            + self.top_words_interpolation_weight * prob)
                    unigram_probs[(word,)] /= (self.top_words_interpolation_weight + self.interpolation_weights[1])

            assert(("</s>",) in unigram_probs)

            tot_prob = 0
            for tup, prob in unigram_probs.iteritems():
                assert(len(tup) == 1)
                word = tup[0]
                if not (self.top_word_probs is not None and word in self.top_word_probs):
                    # Scale, by the unigram interpolation weight, the unigram
                    # probabilities for the words that are not in the top words list
                    prob *= self.interpolation_weights[1]
                    prob /= (self.top_words_interpolation_weight + self.interpolation_weights[1])
                tot_prob += prob

                if word == "</s>":
                    # end-of-sentence; go to the final_state
                    word = 0
                    fst.append((unigram_state, final_state, word, word,
                                -math.log(prob)))
                else:
                    if (word,) in history2state:
                        e = history2state[(word,)]
                        fst.append((unigram_state, e, word, word, -math.log(prob)))
                    else:
                        # There is no state corresponding to this history. So
                        # go directly to the unigram state
                        fst.append((unigram_state, unigram_state, word, word, -math.log(prob)))

        if order == 2:
            # Add backoff arcs
            fst.append((start_state, unigram_state, self.backoff_arc_id, 0, -math.log(1-self.interpolation_weights[2])))
            for s in range(final_state + 1, num_states):
                fst.append((s, unigram_state, self.backoff_arc_id, 0, -math.log(1-self.interpolation_weights[2])))
        elif order == 1:
            fst.append((start_state, unigram_state, 0, 0, 0.0))

        fst.append((final_state, 0))
        fst.sort(key = lambda x:(x[0],x[1]))
        return fst

def PrintFst(utt_id, fst, file_handle):
    print (utt_id, file = file_handle)
    for tup in fst:
        if (len(tup) != 2 and len(tup) != 5):
            raise TypeError("Invalid number of entries in tup {0}".format(str(tup)))

        if len(tup) == 2:
            print ("{0:d} {1:f}".format(tup[0], tup[1]), file = file_handle)
        else:
            print ("{0:d} {1:d} {2:d} {3:d} {4:f}".format(tup[0], tup[1], tup[2], tup[3], tup[4]), file = file_handle)
    print ("", file = file_handle) # Empty line terminates FST in text-archive format


def GetArgs():
    parser = argparse.ArgumentParser(description="""
    Make an FST from the reference transcript.""",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument("--oov", dest='oov_id', type = int,
                        help = "integer for oov word")
    parser.add_argument("--backoff-arc-id", type = int, required = True,
                        help = "integer corresponding to the disambiguation symbol for backoff arcs")

    ngram_fst_parser = parser.add_argument_group('ngram_fst')
    ngram_fst_parser.add_argument("--ngram.order", dest = 'ngram_config.order',
                        type = int, default = 1, choices = [1,2],
                        help = "Order of ngram used")
    ngram_fst_parser.add_argument("--ngram.unigram-interpolation-weight", dest = 'ngram_config.unigram_interpolation_weight',
                        type = float, default = 1.0,
                        help = "interpolation weight for biased unigram LM")
    ngram_fst_parser.add_argument("--ngram.bigram-interpolation-weight", dest = 'ngram_config.bigram_interpolation_weight',
                        type = float, default = 0,
                        help = "interpolation weight for biased bigram LM")

    parser.add_argument("--top-words.list", dest = 'top_words_list', type = str,
                        help="File containing the top words and their "
                        "probabilities to be included in the "
                        "interpolation unigram model")
    parser.add_argument("--top-words.interpolation-weight", dest = 'top_words_interpolation_weight',
                        type = float, default = 1.0,
                        help = "Interpolation weight for the top words unigram "
                        "relative to the utterance-specific LM")

    parser.add_argument("--verbosity", type = int, default = 0,
                        help = "verbosity level")
    parser.add_argument("text_ark", type = str,
                        help = "Archive of text of the utterances for which "
                        "the LMs need to be built")
    parser.add_argument("fsts_ark", type = str,
                        help = "Output archive of utterance fsts")

    if verbose_level > 2:
        print(' '.join(sys.argv), file = sys.stderr)

    args = parser.parse_args()

    ngram_options = [action.dest for action in ngram_fst_parser._group_actions]

    d = vars(args)

    for name in ngram_options:
        group,dest = name.split('.',2)
        d.setdefault(group, argparse.Namespace())
        group_vars = vars(d[group])
        group_vars[dest] = d[name]

    args = CheckArgs(args)

    return args

def CheckArgs(args):

    if args.ngram_config.order <= 0:
        raise ValueError("ngram-fst.order must be > 0")

    args.ngram_config.interpolation_weights = [0]

    args.ngram_config.interpolation_weights.append(args.ngram_config.unigram_interpolation_weight)

    if args.ngram_config.order == 2:
        args.ngram_config.interpolation_weights.append(args.ngram_config.bigram_interpolation_weight)

    assert (len(args.ngram_config.interpolation_weights) == (args.ngram_config.order + 1))

    if args.top_words_interpolation_weight <= 0:
        raise ValueError("top-word-interpolation-weight must be > 0")

    if args.text_ark == "-":
        args.text_ark_handle = sys.stdin
    else:
        args.text_ark_handle = open(args.text_ark)

    if args.fsts_ark == "-":
        args.fsts_ark_handle = sys.stdout
    else:
        args.fsts_ark_handle = open(args.fsts_ark, 'w')

    verbose_level = args.verbosity

    return args

def Main():
    args = GetArgs()

    top_word_probs = dict()

    if args.top_words_list is not None:
        total_prob = 0
        for line in open(args.top_words_list):
            splits = line.strip().split()
            if len(splits) != 2:
                raise TypeError("Unexpected format of line {0}; {1} must have two columns <probability> <word>".format(line.strip(), args.top_words_list))

            prob = float(splits[0])
            if prob < 0 or prob > 1:
                raise ValueError("First column in {0} must be a probability with value [0,1]; got {1}".format(args.top_words_list, prob))

            total_prob += prob

            word = int(splits[1])

            top_word_probs[word] = top_word_probs.setdefault(word, 0) + prob

        if total_prob > 1:
            raise ValueError("Sum of top word probabilities must be < 1")
        top_word_oov_prob = 1 - total_prob

        # Add the missing probability in the top words list to the
        # OOV probability
        top_word_probs[args.oov_id] = (top_word_probs.setdefault(args.oov_id, 0)
                                                + top_word_oov_prob)

    if args.ngram_config.order > 2:
        raise NotImplementedError("--ngram.order must be <= 2")

    ngrams = Ngrams(args.oov_id, args.backoff_arc_id,
                    args.ngram_config.order,
                    args.ngram_config.interpolation_weights,
                    top_word_probs if args.top_words_list is not None else None,
                    args.top_words_interpolation_weight)

    for line in args.text_ark_handle:
        splits = line.strip().split()
        utt_id = splits[0]

        ngrams.ResetStats()

        # w_1 w_2 ... w_i ... w_j ...
        for j in range(1,len(splits)):
            i = j - args.ngram_config.order + 1

            tup = ()
            if i < 1:
                # Add beginning-of-sentence to create history of required length
                tup = ("<s>",) * (1 - i)
                i = 1
            tup += tuple([ int(x) for x in splits[i:(j+1)] ])

            assert(len(tup) == args.ngram_config.order)

            # Update all ngram counts
            ngrams.AddStats(tup)

        i = len(splits) - args.ngram_config.order + 1

        tup = ()
        if i < 1:
            # Add beginning-of-sentence to create history of required length
            tup = ("<s>",) * (1 - i)
            i = 1
        if i < len(splits):
            tup += tuple([ int(x) for x in splits[i:] ])
        tup += ("</s>",)  # end-of-sentence

        # Update all ngram counts
        ngrams.AddStats(tup)

        if verbose_level > 2:
            print("ngrams: " + str(ngrams.stats), file = sys.stderr)

        # list of probs for different order ngrams
        probs = [ dict() for o in range(args.ngram_config.order+1) ]

        # Assume we only need the highest order ngram
        probs[1] = ngrams.GetProbs(1)
        if args.ngram_config.order > 1:
            probs[args.ngram_config.order] = ngrams.GetProbs(args.ngram_config.order)

        fst = ngrams.GetFst(args.ngram_config.order, probs)

        # Print FST for the utterance
        PrintFst(utt_id, fst, args.fsts_ark_handle)

if __name__ == "__main__":
    Main()
