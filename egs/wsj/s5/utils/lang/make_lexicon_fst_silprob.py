#!/usr/bin/env python3
# Copyright   2018  Johns Hopkins University (author: Daniel Povey)
#             2018  Jiedan Zhu
# Apache 2.0.
# see get_args() below for usage message.

import argparse
import os
import sys
import math

# The use of latin-1 encoding does not preclude reading utf-8.  latin-1
# encoding means "treat words as sequences of bytes", and it is compatible
# with utf-8 encoding as well as other encodings such as gbk, as long as the
# spaces are also spaces in ascii (which we check).  It is basically how we
# emulate the behavior of python before python3.

sys.stdout = open(1, 'w', encoding='latin-1', closefd=False)
sys.stderr = open(2, 'w', encoding='latin-1', closefd=False)


def get_args():
    parser = argparse.ArgumentParser(description="""This script creates the
       text form of a lexicon FST, to be compiled by fstcompile using the
       appropriate symbol tables (phones.txt and words.txt) .  It will mostly
       be invoked indirectly via utils/prepare_lang.sh.  The output goes to
       the stdout.""")

    parser.add_argument('--sil-phone', dest='sil_phone', type=str,
                        help="""Text form of optional-silence phone, e.g. 'SIL'.  See also
                        the --silprob option.""")
    parser.add_argument('--sil-disambig', dest='sil_disambig', type=str, default="<eps>",
                        help="""Disambiguation symbol to disambiguate silence, e.g. #5.
                        Will only be supplied if you are creating the version of L.fst
                        with disambiguation symbols, intended for use with cyclic G.fst.
                        This symbol was introduced to fix a rather obscure source of
                        nondeterminism of CLG.fst, that has to do with reordering of
                        disambiguation symbols and phone symbols.""")
    parser.add_argument('lexiconp', type=str,
                        help="""Filename of lexicon with pronunciation probabilities
                        (normally lexiconp.txt), with lines of the form
                        'word pron-prob prob-of-sil correction-term-for-sil correction-term-for-no-sil p1 p2...',
                        e.g. 'a   1.0  0.8 1.2  0.6  ay'""")
    parser.add_argument('silprobs', type=str,
                        help="""Filename with silence probabilities, with lines of the form
                        '<s> p(sil-after|<s>) //
                        </s>_s correction-term-for-sil-for-</s> //
                        </s>_n correction-term-for-no-sil-for-</s> //
                        overall p(overall-sil), where // represents line break.
                        See also utils/dict_dir_add_pronprobs.sh,
                        which creates this file as silprob.txt.""")
    args = parser.parse_args()
    return args


def read_silprobs(filename):
    """ Reads the silprobs file (e.g. silprobs.txt) which will have a format like this:
     <s> 0.99
     </s>_s 2.50607106867326
     </s>_n 0.00653829808100956
     overall 0.20
    and returns it as a 4-tuple, e.g. in this example (0.99, 2.50, 0.006, 0.20)
    """
    silbeginprob = -1
    silendcorrection = -1
    nonsilendcorrection = -1
    siloverallprob = -1
    with open(filename, 'r', encoding='latin-1') as f:
        for line in f:
            a = line.split()
            if len(a) != 2:
                print("{0}: error: found bad line '{1}' in silprobs file {1} ".format(
                    sys.argv[0], line.strip(), filename), file=sys.stderr)
                sys.exit(1)
            label = a[0]
            try:
                if label == "<s>":
                    silbeginprob = float(a[1])
                elif label == "</s>_s":
                    silendcorrection = float(a[1])
                elif label == "</s>_n":
                    nonsilendcorrection = float(a[1])
                elif label == "overall":
                    siloverallprob = float(a[1]) # this is not in use, still keep it?
                else:
                    raise RuntimeError()
            except:
                print("{0}: error: found bad line '{1}' in silprobs file {1}"
                      .format(sys.argv[0], line.strip(), filename),
                      file=sys.stderr)
                sys.exit(1)
    if (silbeginprob <= 0.0 or silbeginprob > 1.0 or
        silendcorrection <= 0.0 or nonsilendcorrection <= 0.0 or
        siloverallprob <= 0.0 or siloverallprob > 1.0):
        print("{0}: error: prob is not correct in silprobs file {1}."
            .format(sys.argv[0], filename), file=sys.stderr)
        sys.exit(1)
    return (silbeginprob, silendcorrection, nonsilendcorrection, siloverallprob)


def read_lexiconp(filename):
    """Reads the lexiconp.txt file in 'filename', with lines like
    'word p(pronunciation|word) p(sil-after|word) correction-term-for-sil
    correction-term-for-no-sil p1 p2 ...'.
    Returns a list of tuples (word, pron_prob, word_sil_prob,
    sil_word_correction, non_sil_word_correction, prons), where 'word' is a string,
   'pron_prob', a float, is the pronunciation probability (which must be >0.0
    and would normally be <=1.0), 'word_sil_prob' is a float,
    'sil_word_correction' is a float, 'non_sil_word_correction' is a float,
    and 'pron' is a list of strings representing phones.
    An element in the returned list might be
    ('hello', 1.0, 0.5, 0.3, 0.6, ['h', 'eh', 'l', 'ow']).
    """
    ans = []
    found_empty_prons = False
    found_large_pronprobs = False
    # See the comment near the top of this file, RE why we use latin-1.
    with open(filename, 'r', encoding='latin-1') as f:
        for line in f:
            a = line.split()
            if len(a) < 2:
                print("{0}: error: found bad line '{1}' in lexicon file {1} ".format(
                    sys.argv[0], line.strip(), filename), file=sys.stderr)
                sys.exit(1)
            word = a[0]
            if word == "<eps>":
                # This would clash with the epsilon symbol normally used in OpenFst.
                print("{0}: error: found <eps> as a word in lexicon file "
                      "{1}".format(line.strip(), filename), file=sys.stderr)
                sys.exit(1)
            try:
                pron_prob = float(a[1])
                word_sil_prob = float(a[2])
                sil_word_correction = float(a[3])
                non_sil_word_correction = float(a[4])
            except:
                print("{0}: error: found bad line '{1}' in lexicon file {2}, 2nd field "
                      "through 5th field should be numbers".format(sys.argv[0],
                                                                   line.strip(), filename),
                      file=sys.stderr)
                sys.exit(1)
            prons = a[5:]
            if pron_prob <= 0.0:
                print("{0}: error: invalid pron-prob in line '{1}' of lexicon file {2} ".format(
                    sys.argv[0], line.strip(), filename), file=sys.stderr)
                sys.exit(1)
            if len(prons) == 0:
                found_empty_prons = True
            ans.append((
                word, pron_prob, word_sil_prob,
                sil_word_correction, non_sil_word_correction, prons))
            if pron_prob > 1.0:
                found_large_pronprobs = True
    if found_empty_prons:
        print("{0}: warning: found at least one word with an empty pronunciation "
              "in lexicon file {1}.".format(sys.argv[0], filename),
              file=sys.stderr)
    if found_large_pronprobs:
        print("{0}: warning: found at least one word with pron-prob >1.0 "
              "in {1}".format(sys.argv[0], filename), file=sys.stderr)
    if len(ans) == 0:
        print("{0}: error: found no pronunciations in lexicon file {1}".format(
            sys.argv[0], filename), file=sys.stderr)
        sys.exit(1)
    return ans


def write_fst_with_silence(lexicon, silprobs, sil_phone, sil_disambig):
    """Writes the text format of L.fst to the standard output.  This version is for
       when --sil-prob != 0.0, meaning there is optional silence
     'lexicon' is a list of 5-tuples
     (word, pronprob, wordsilprob, silwordcorrection, nonsilwordcorrection, pron)
         as returned by read_lexiconp().
     'silprobs' is a 4-tuple of probabilities as returned by read_silprobs().
     'sil_phone' is the silence phone, e.g. "SIL".
     'sil_disambig' is either '<eps>', or the silence disambiguation symbol, e.g. "#5".
    """
    silbeginprob, silendcorrection, nonsilendcorrection, siloverallprob = silprobs
    sil_cost = -math.log(silbeginprob)
    no_sil_cost = -math.log(1.0 - silbeginprob);
    sil_end_correction_cost = -math.log(silendcorrection)
    non_sil_end_correction_cost = -math.log(nonsilendcorrection);
    start_state = 0
    non_sil_state = 1  # words enter and leave from here
    sil_state = 2   # words terminate here when followed by silence; this state
                    # has a silence transition to loop_state.
    next_state = 3  # the next un-allocated state, will be incremented as we go.

    # Arcs from the start state to the silence and nonsilence loop states
    # The one to the nonsilence state has the silence disambiguation symbol
    # (We always use that symbol on the *non*-silence-containing arcs, which
    # avoids having to introduce extra arcs).
    print('{src}\t{dest}\t{phone}\t{word}\t{cost}'.format(
        src=start_state, dest=non_sil_state,
        phone=sil_phone, word='<eps>', cost=no_sil_cost))
    print('{src}\t{dest}\t{phone}\t{word}\t{cost}'.format(
        src=start_state, dest=sil_state,
        phone=sil_disambig, word='<eps>', cost=sil_cost))

    for (word, pronprob, wordsilprob, silwordcorrection, nonsilwordcorrection, pron) in lexicon:
        pron_cost = -math.log(pronprob)
        word_to_sil_cost = -math.log(wordsilprob)
        word_to_non_sil_cost = -math.log(1.0 - wordsilprob)
        sil_to_word_cost = -math.log(silwordcorrection)
        non_sil_to_word_cost = -math.log(nonsilwordcorrection)

        if len(pron) == 0:
            # this is not really expected but we try to handle it gracefully.
            pron = ['<eps>']

        new_state = next_state  # allocate a new state
        next_state += 1
        # Create transitions from both non_sil_state and sil_state to 'new_state',
        # with the word label and the word's first phone on them
        print("{src}\t{dest}\t{phone}\t{word}\t{cost}".format(
            src=non_sil_state, dest=new_state,
            phone=pron[0], word=word, cost=(pron_cost + non_sil_to_word_cost)))
        print("{src}\t{dest}\t{phone}\t{word}\t{cost}".format(
            src=sil_state, dest=new_state,
            phone=pron[0], word=word, cost=(pron_cost + sil_to_word_cost)))
        cur_state = new_state

        # it's then linear till the end of the word
        for i in range(1, len(pron)):
            new_state = next_state
            next_state += 1
            print("{src}\t{dest}\t{phone}\t<eps>".format(
                src=cur_state, dest=new_state, phone=pron[i]))
            cur_state = new_state

        # ... and from there we return via two arcs to the silence and
        # nonsilence state.  the silence-disambig symbol, if used,q
        # goes on the nonsilence arc; this saves us having to insert an epsilon.
        print("{src}\t{dest}\t{phone}\t{word}\t{cost}".format(
            src=cur_state,  dest=non_sil_state,
            phone=sil_disambig, word='<eps>',
            cost=word_to_non_sil_cost))
        print("{src}\t{dest}\t{phone}\t{word}\t{cost}".format(
            src=cur_state, dest=sil_state,
            phone=sil_phone, word='<eps>',
            cost=word_to_sil_cost))

    print('{src}\t{cost}'.format(src=sil_state, cost=sil_end_correction_cost))
    print('{src}\t{cost}'.format(src=non_sil_state, cost=non_sil_end_correction_cost))


def main():
    args = get_args()
    silprobs = read_silprobs(args.silprobs)
    lexicon = read_lexiconp(args.lexiconp)
    write_fst_with_silence(lexicon, silprobs, args.sil_phone, args.sil_disambig)


if __name__ == '__main__':
      main()
