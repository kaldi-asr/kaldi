#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Copyright 2021  Johns Hopkins University (Author: Ruizhe Huang)
# Apache 2.0.

# This is an implementation of ``Entropy-based Pruning of Backoff Language Models''
# in the same way as SRILM.

################################################
#        Useful links/References:
################################################
# https://github.com/BitSpeech/SRILM/blob/d571a4424fb0cf08b29fbfccfddd092ea969eae3/lm/src/NgramLM.cc#L2330
# https://github.com/BitSpeech/SRILM/blob/d571a4424fb0cf08b29fbfccfddd092ea969eae3/lm/src/NgramLM.cc#L2124
# https://github.com/BitSpeech/SRILM/blob/d571a4424fb0cf08b29fbfccfddd092ea969eae3/lm/src/LM.cc#L527
# https://github.com/BitSpeech/SRILM/blob/d571a4424fb0cf08b29fbfccfddd092ea969eae3/flm/src/FNgramLM.cc#L2124
# https://github.com/sfischer13/python-arpa

################################################
#                How to use:
################################################
# python3 ngram_entropy_pruning.py -threshold $threshold -lm $input_lm -write-lm $pruned_lm

################################################
#             SRILM commands:
################################################
# to_prune_lm=egs/swbd/s5c/data/local/lm/sw1.o3g.kn.gz
# vocab=egs/swbd/s5c/data/local/lm/wordlist
# order=3
# oov_symbol="<unk>"
# threshold=4.7e-5
# pruned_lm=temp.${threshold}.gz
# ngram -unk -map-unk "$oov_symbol" -vocab $vocab -order $order -prune ${threshold} -lm ${to_prune_lm} -write-lm ${pruned_lm}
#
# lm=
# ngram -unk -lm $lm -ppl heldout
# ngram -unk -lm $lm -ppl heldout -debug 3

import argparse
import logging
import math

import gzip
from io import StringIO
from collections import OrderedDict
from collections import defaultdict
from enum import Enum, unique
import re

parser = argparse.ArgumentParser(description="""
    Prune an n-gram language model based on the relative entropy 
    between the original and the pruned model, based on Andreas Stolcke's paper.
    An n-gram entry is removed, if the removal causes (training set) perplexity 
    of the model to increase by less than threshold relative.
    
    The command takes an arpa file and a pruning threshold as input, 
    and outputs a pruned arpa file.
    """)
parser.add_argument("-threshold",
                    type=float,
                    default=1e-6,
                    help="Order of n-gram")
parser.add_argument("-lm",
                    type=str,
                    default=None,
                    help="Path to the input arpa file")
parser.add_argument("-write-lm",
                    type=str,
                    default=None,
                    help="Path to output arpa file after pruning")
parser.add_argument("-minorder",
                    type=int,
                    default=1,
                    help="The minorder parameter limits pruning to "
                    "ngrams of that length and above.")
parser.add_argument("-encoding",
                    type=str,
                    default="utf-8",
                    help="Encoding of the arpa file")
parser.add_argument("-verbose",
                    type=int,
                    default=2,
                    choices=[0, 1, 2, 3, 4, 5],
                    help="Verbose level, where "
                    "0 is most noisy; "
                    "5 is most silent")
args = parser.parse_args()

default_encoding = args.encoding
logging.basicConfig(
    format=
    "%(asctime)s — %(levelname)s — %(funcName)s:%(lineno)d — %(message)s",
    level=args.verbose * 10)


class Context(dict):
    """
    This class stores data for a context h.
    It behaves like a python dict object, except that it has several
    additional attributes.
    """
    def __init__(self):
        super().__init__()
        self.log_bo = None


class Arpa:
    """
    This is a class that implement the data structure of an APRA LM.
    It (as well as some other classes) is modified based on the library
    by Stefan Fischer:
    https://github.com/sfischer13/python-arpa
    """

    UNK = '<unk>'
    SOS = '<s>'
    EOS = '</s>'
    FLOAT_NDIGITS = 7
    base = 10

    @staticmethod
    def _check_input(my_input):
        if not my_input:
            raise ValueError
        elif isinstance(my_input, tuple):
            return my_input
        elif isinstance(my_input, list):
            return tuple(my_input)
        elif isinstance(my_input, str):
            return tuple(my_input.strip().split(' '))
        else:
            raise ValueError

    @staticmethod
    def _check_word(input_word):
        if not isinstance(input_word, str):
            raise ValueError
        if ' ' in input_word:
            raise ValueError

    def _replace_unks(self, words):
        return tuple((w if w in self else self._unk) for w in words)

    def __init__(self, path=None, encoding=None, unk=None):
        self._counts = OrderedDict()
        self._ngrams = OrderedDict(
        )  # Use self._ngrams[len(h)][h][w] for saving the entry of (h,w)
        self._vocabulary = set()
        if unk is None:
            self._unk = self.UNK

        if path is not None:
            self.loadf(path, encoding)

    def __contains__(self, ngram):
        h = ngram[:-1]  # h is a tuple
        w = ngram[-1]  # w is a string/word
        return h in self._ngrams[len(h)] and w in self._ngrams[len(h)][h]

    def contains_word(self, word):
        self._check_word(word)
        return word in self._vocabulary

    def add_count(self, order, count):
        self._counts[order] = count
        self._ngrams[order - 1] = defaultdict(Context)

    def update_counts(self):
        for order in range(1, self.order() + 1):
            count = sum(
                [len(wlist) for _, wlist in self._ngrams[order - 1].items()])
            if count > 0:
                self._counts[order] = count

    def add_entry(self, ngram, p, bo=None, order=None):
        # Note: ngram is a tuple of strings, e.g. ("w1", "w2", "w3")
        h = ngram[:-1]  # h is a tuple
        w = ngram[-1]  # w is a string/word

        # Note that p and bo here are in fact in the log domain (self.base = 10)
        h_context = self._ngrams[len(h)][h]
        h_context[w] = p
        if bo is not None:
            self._ngrams[len(ngram)][ngram].log_bo = bo

        for word in ngram:
            self._vocabulary.add(word)

    def counts(self):
        return sorted(self._counts.items())

    def order(self):
        return max(self._counts.keys(), default=None)

    def vocabulary(self, sort=True):
        if sort:
            return sorted(self._vocabulary)
        else:
            return self._vocabulary

    def _entries(self, order):
        return (self._entry(h, w)
                for h, wlist in self._ngrams[order - 1].items() for w in wlist)

    def _entry(self, h, w):
        # return the entry for the ngram (h, w)
        ngram = h + (w, )
        log_p = self._ngrams[len(h)][h][w]
        log_bo = self._log_bo(ngram)
        if log_bo is not None:
            return round(log_p, self.FLOAT_NDIGITS), ngram, round(
                log_bo, self.FLOAT_NDIGITS)
        else:
            return round(log_p, self.FLOAT_NDIGITS), ngram

    def _log_bo(self, ngram):
        if len(ngram) in self._ngrams and ngram in self._ngrams[len(ngram)]:
            return self._ngrams[len(ngram)][ngram].log_bo
        else:
            return None

    def _log_p(self, ngram):
        h = ngram[:-1]  # h is a tuple
        w = ngram[-1]  # w is a string/word
        if h in self._ngrams[len(h)] and w in self._ngrams[len(h)][h]:
            return self._ngrams[len(h)][h][w]
        else:
            return None

    def log_p_raw(self, ngram):
        log_p = self._log_p(ngram)
        if log_p is not None:
            return log_p
        else:
            if len(ngram) == 1:
                raise KeyError
            else:
                log_bo = self._log_bo(ngram[:-1])
                if log_bo is None:
                    log_bo = 0
                return log_bo + self.log_p_raw(ngram[1:])

    def log_joint_prob(self, sequence):
        # Compute the joint prob of the sequence based on the chain rule
        # Note that sequence should be a tuple of strings
        #
        # Reference:
        # https://github.com/BitSpeech/SRILM/blob/d571a4424fb0cf08b29fbfccfddd092ea969eae3/lm/src/LM.cc#L527

        log_joint_p = 0
        seq = sequence
        while len(seq) > 0:
            log_joint_p += self.log_p_raw(seq)
            seq = seq[:-1]

            # If we're computing the marginal probability of the unigram
            # <s> context we have to look up </s> instead since the former
            # has prob = 0.
            if len(seq) == 1 and seq[0] == self.SOS:
                seq = (self.EOS, )

        return log_joint_p

    def set_new_context(self, h):
        old_context = self._ngrams[len(h)][h]
        self._ngrams[len(h)][h] = Context()
        return old_context

    def log_p(self, ngram):
        words = self._check_input(ngram)
        if self._unk:
            words = self._replace_unks(words)
        return self.log_p_raw(words)

    def log_s(self, sentence, sos=SOS, eos=EOS):
        words = self._check_input(sentence)
        if self._unk:
            words = self._replace_unks(words)
        if sos:
            words = (sos, ) + words
        if eos:
            words = words + (eos, )
        result = sum(
            self.log_p_raw(words[:i]) for i in range(1,
                                                     len(words) + 1))
        if sos:
            result = result - self.log_p_raw(words[:1])
        return result

    def p(self, ngram):
        return self.base**self.log_p(ngram)

    def s(self, sentence):
        return self.base**self.log_s(sentence)

    def write(self, fp):
        fp.write('\n\\data\\\n')
        for order, count in self.counts():
            fp.write('ngram {}={}\n'.format(order, count))
        fp.write('\n')
        for order, _ in self.counts():
            fp.write('\\{}-grams:\n'.format(order))
            for e in self._entries(order):
                prob = e[0]
                ngram = ' '.join(e[1])
                if len(e) == 2:
                    fp.write('{}\t{}\n'.format(prob, ngram))
                elif len(e) == 3:
                    backoff = e[2]
                    fp.write('{}\t{}\t{}\n'.format(prob, ngram, backoff))
                else:
                    raise ValueError
            fp.write('\n')
        fp.write('\\end\\\n')


class ArpaParser:
    """
    This is a class that implement a parser of an arpa file
    """
    @unique
    class State(Enum):
        DATA = 1
        COUNT = 2
        HEADER = 3
        ENTRY = 4

    re_count = re.compile(r'^ngram (\d+)=(\d+)$')
    re_header = re.compile(r'^\\(\d+)-grams:$')
    re_entry = re.compile('^(-?\\d+(\\.\\d+)?([eE]-?\\d+)?)'
                          '\t'
                          '(\\S+( \\S+)*)'
                          '(\t((-?\\d+(\\.\\d+)?)([eE]-?\\d+)?))?$')

    def _parse(self, fp):
        self._result = []
        self._state = self.State.DATA
        self._tmp_model = None
        self._tmp_order = None
        for line in fp:
            line = line.strip()
            if self._state == self.State.DATA:
                self._data(line)
            elif self._state == self.State.COUNT:
                self._count(line)
            elif self._state == self.State.HEADER:
                self._header(line)
            elif self._state == self.State.ENTRY:
                self._entry(line)
        if self._state != self.State.DATA:
            raise Exception(line)
        return self._result

    def _data(self, line):
        if line == '\\data\\':
            self._state = self.State.COUNT
            self._tmp_model = Arpa()
        else:
            pass  # skip comment line

    def _count(self, line):
        match = self.re_count.match(line)
        if match:
            order = match.group(1)
            count = match.group(2)
            self._tmp_model.add_count(int(order), int(count))
        elif not line:
            self._state = self.State.HEADER  # there are no counts
        else:
            raise Exception(line)

    def _header(self, line):
        match = self.re_header.match(line)
        if match:
            self._state = self.State.ENTRY
            self._tmp_order = int(match.group(1))
        elif line == '\\end\\':
            self._result.append(self._tmp_model)
            self._state = self.State.DATA
            self._tmp_model = None
            self._tmp_order = None
        elif not line:
            pass  # skip empty line
        else:
            raise Exception(line)

    def _entry(self, line):
        match = self.re_entry.match(line)
        if match:
            p = self._float_or_int(match.group(1))
            ngram = tuple(match.group(4).split(' '))
            bo_match = match.group(7)
            bo = self._float_or_int(bo_match) if bo_match else None
            self._tmp_model.add_entry(ngram, p, bo, self._tmp_order)
        elif not line:
            self._state = self.State.HEADER  # last entry
        else:
            raise Exception(line)

    @staticmethod
    def _float_or_int(s):
        f = float(s)
        i = int(f)
        if str(i) == s:  # don't drop trailing ".0"
            return i
        else:
            return f

    def load(self, fp):
        """Deserialize fp (a file-like object) to a Python object."""
        return self._parse(fp)

    def loadf(self, path, encoding=None):
        """Deserialize path (.arpa, .gz) to a Python object."""
        path = str(path)
        if path.endswith('.gz'):
            with gzip.open(path, mode='rt', encoding=encoding) as f:
                return self.load(f)
        else:
            with open(path, mode='rt', encoding=encoding) as f:
                return self.load(f)

    def loads(self, s):
        """Deserialize s (a str) to a Python object."""
        with StringIO(s) as f:
            return self.load(f)

    def dump(self, obj, fp):
        """Serialize obj to fp (a file-like object) in ARPA format."""
        obj.write(fp)

    def dumpf(self, obj, path, encoding=None):
        """Serialize obj to path in ARPA format (.arpa, .gz)."""
        path = str(path)
        if path.endswith('.gz'):
            with gzip.open(path, mode='wt', encoding=encoding) as f:
                return self.dump(obj, f)
        else:
            with open(path, mode='wt', encoding=encoding) as f:
                self.dump(obj, f)

    def dumps(self, obj):
        """Serialize obj to an ARPA formatted str."""
        with StringIO() as f:
            self.dump(obj, f)
            return f.getvalue()


def add_log_p(prev_log_sum, log_p, base):
    return math.log(base**log_p + base**prev_log_sum, base)


def compute_numerator_denominator(lm, h):
    log_sum_seen_h = -math.inf
    log_sum_seen_h_lower = -math.inf
    base = lm.base
    for w, log_p in lm._ngrams[len(h)][h].items():
        log_sum_seen_h = add_log_p(log_sum_seen_h, log_p, base)

        ngram = h + (w, )
        log_p_lower = lm.log_p_raw(ngram[1:])
        log_sum_seen_h_lower = add_log_p(log_sum_seen_h_lower, log_p_lower,
                                         base)

    numerator = 1.0 - base**log_sum_seen_h
    denominator = 1.0 - base**log_sum_seen_h_lower
    return numerator, denominator


def prune(lm, threshold, minorder):
    # Reference:
    # https://github.com/BitSpeech/SRILM/blob/d571a4424fb0cf08b29fbfccfddd092ea969eae3/lm/src/NgramLM.cc#L2330

    for i in range(lm.order(), max(minorder - 1, 1),
                   -1):  # i is the order of the ngram (h, w)
        logging.info("processing %d-grams ..." % i)
        count_pruned_ngrams = 0

        h_dict = lm._ngrams[i - 1]
        for h in list(h_dict.keys()):
            # old backoff weight, BOW(h)
            log_bow = lm._log_bo(h)
            if log_bow is None:
                log_bow = 0

            # Compute numerator and denominator of the backoff weight,
            # so that we can quickly compute the BOW adjustment due to
            # leaving out one prob.
            numerator, denominator = compute_numerator_denominator(lm, h)

            # assert abs(math.log(numerator, lm.base) - math.log(denominator, lm.base) - h_dict[h].log_bo) < 1e-5

            # Compute the marginal probability of the context, P(h)
            h_log_p = lm.log_joint_prob(h)

            all_pruned = True
            pruned_w_set = set()

            for w, log_p in h_dict[h].items():
                ngram = h + (w, )

                # lower-order estimate for ngramProb, P(w|h')
                backoff_prob = lm.log_p_raw(ngram[1:])

                # Compute BOW after removing ngram, BOW'(h)
                new_log_bow = math.log(numerator + lm.base ** log_p, lm.base) - \
                              math.log(denominator + lm.base ** backoff_prob, lm.base)

                # Compute change in entropy due to removal of ngram
                delta_prob = backoff_prob + new_log_bow - log_p
                delta_entropy = - (lm.base ** h_log_p) * \
                                ((lm.base ** log_p) * delta_prob +
                                 numerator * (new_log_bow - log_bow))

                # compute relative change in model (training set) perplexity
                perp_change = lm.base**delta_entropy - 1.0

                pruned = threshold > 0 and perp_change < threshold

                # Make sure we don't prune ngrams whose backoff nodes are needed
                if pruned and \
                        len(ngram) in lm._ngrams and \
                        len(lm._ngrams[len(ngram)][ngram]) > 0:
                    pruned = False

                logging.debug("CONTEXT " + str(h) + " WORD " + w +
                              " CONTEXTPROB %f " % h_log_p +
                              " OLDPROB %f " % log_p + " NEWPROB %f " %
                              (backoff_prob + new_log_bow) +
                              " DELTA-H %f " % delta_entropy +
                              " DELTA-LOGP %f " % delta_prob +
                              " PPL-CHANGE %f " % perp_change + " PRUNED " +
                              str(pruned))

                if pruned:
                    pruned_w_set.add(w)
                    count_pruned_ngrams += 1
                else:
                    all_pruned = False

            # If we removed all ngrams for this context we can
            # remove the context itself, but only if the present
            # context is not a prefix to a longer one.
            if all_pruned and len(pruned_w_set) == len(h_dict[h]):
                del h_dict[
                    h]  # this context h is no longer needed, as its ngram prob is stored at its own context h'
            elif len(pruned_w_set) > 0:
                # The pruning for this context h is actually done here
                old_context = lm.set_new_context(h)

                for w, p_w in old_context.items():
                    if w not in pruned_w_set:
                        lm.add_entry(
                            h + (w, ),
                            p_w)  # the entry hw is stored at the context h

                # We need to recompute the back-off weight, but
                # this can only be done after completing the pruning
                # of the lower-order ngrams.
                # Reference:
                # https://github.com/BitSpeech/SRILM/blob/d571a4424fb0cf08b29fbfccfddd092ea969eae3/flm/src/FNgramLM.cc#L2124

        logging.info("pruned %d %d-grams" % (count_pruned_ngrams, i))

    # recompute backoff weights
    for i in range(max(minorder - 1, 1) + 1,
                   lm.order() +
                   1):  # be careful of this order: from low- to high-order
        for h in lm._ngrams[i - 1]:
            numerator, denominator = compute_numerator_denominator(lm, h)
            new_log_bow = math.log(numerator, lm.base) - math.log(
                denominator, lm.base)
            lm._ngrams[len(h)][h].log_bo = new_log_bow

    # update counts
    lm.update_counts()

    return


def check_h_is_valid(lm, h):
    sum_under_h = sum(
        [lm.base**lm.log_p_raw(h + (w, )) for w in lm.vocabulary(sort=False)])
    if abs(sum_under_h - 1.0) > 1e-6:
        logging.info("warning: %s %f" % (str(h), sum_under_h))
        return False
    else:
        return True


def validate_lm(lm):
    # sanity check if the conditional probability sums to one under each context h
    for i in range(lm.order(), 0, -1):  # i is the order of the ngram (h, w)
        logging.info("validating %d-grams ..." % i)
        h_dict = lm._ngrams[i - 1]
        for h in h_dict.keys():
            check_h_is_valid(lm, h)


def compare_two_apras(path1, path2):
    pass


if __name__ == '__main__':
    # load an arpa file
    logging.info("Loading the arpa file from %s" % args.lm)
    parser = ArpaParser()
    models = parser.loadf(args.lm, encoding=default_encoding)
    lm = models[0]  # ARPA files may contain several models.
    logging.info("Stats before pruning:")
    for i, cnt in lm.counts():
        logging.info("ngram %d=%d" % (i, cnt))

    # prune it, the language model will be modified in-place
    logging.info("Start pruning the model with threshold=%.3E..." %
                 args.threshold)
    prune(lm, args.threshold, args.minorder)

    # validate_lm(lm)

    # write the arpa language model to a file
    logging.info("Stats after pruning:")
    for i, cnt in lm.counts():
        logging.info("ngram %d=%d" % (i, cnt))
    logging.info("Saving the pruned arpa file to %s" % args.write_lm)
    parser.dumpf(lm, args.write_lm, encoding=default_encoding)
    logging.info("Done.")
