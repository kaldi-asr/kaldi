#!/usr/bin/env python3
# Copyright   2018  Johns Hopkins University (author: Daniel Povey)
#             2018  Jiedan Zhu
# Apache 2.0.
# see get_args() below for usage message.

import argparse
import os
import sys
import math
import re

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
       the stdout.

       This version is for a lexicon with word-specific silence probabilities,
       see http://www.danielpovey.com/files/2015_interspeech_silprob.pdf
       for an explanation""")

    parser.add_argument('--sil-phone', dest='sil_phone', type=str,
                        help="Text form of optional-silence phone, e.g. 'SIL'.")
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
    parser.add_argument('--left-context-phones', dest='left_context_phones', type=str,
                        help="""Only relevant if --nonterminals is also supplied; this relates
                        to grammar decoding (see http://kaldi-asr.org/doc/grammar.html or
                        src/doc/grammar.dox).  Format is a list of left-context phones,
                        in text form, one per line.  E.g. data/lang/phones/left_context_phones.txt""")
    parser.add_argument('--nonterminals', type=str,
                        help="""If supplied, --left-context-phones must also be supplied.
                        List of user-defined nonterminal symbols such as #nonterm:contact_list,
                        one per line.  E.g. data/local/dict/nonterminals.txt.""")

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
        whitespace = re.compile("[ \t]+")
        for line in f:
            a = whitespace.split(line.strip(" \t\r\n"))
            if len(a) != 2:
                print("{0}: error: found bad line '{1}' in silprobs file {1} ".format(
                    sys.argv[0], line.strip(" \t\r\n"), filename), file=sys.stderr)
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
                      .format(sys.argv[0], line.strip(" \t\r\n"), filename),
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
    whitespace = re.compile("[ \t]+")
    with open(filename, 'r', encoding='latin-1') as f:
        for line in f:
            a = whitespace.split(line.strip(" \t\r\n"))
            if len(a) < 2:
                print("{0}: error: found bad line '{1}' in lexicon file {1} ".format(
                    sys.argv[0], line.strip(" \t\r\n"), filename), file=sys.stderr)
                sys.exit(1)
            word = a[0]
            if word == "<eps>":
                # This would clash with the epsilon symbol normally used in OpenFst.
                print("{0}: error: found <eps> as a word in lexicon file "
                      "{1}".format(line.strip(" \t\r\n"), filename), file=sys.stderr)
                sys.exit(1)
            try:
                pron_prob = float(a[1])
                word_sil_prob = float(a[2])
                sil_word_correction = float(a[3])
                non_sil_word_correction = float(a[4])
            except:
                print("{0}: error: found bad line '{1}' in lexicon file {2}, 2nd field "
                      "through 5th field should be numbers".format(sys.argv[0],
                                                                   line.strip(" \t\r\n"), filename),
                      file=sys.stderr)
                sys.exit(1)
            prons = a[5:]
            if pron_prob <= 0.0:
                print("{0}: error: invalid pron-prob in line '{1}' of lexicon file {2} ".format(
                    sys.argv[0], line.strip(" \t\r\n"), filename), file=sys.stderr)
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


def write_nonterminal_arcs(start_state, sil_state, non_sil_state,
                           next_state, sil_phone,
                           nonterminals, left_context_phones):
    """This function relates to the grammar-decoding setup, see
    kaldi-asr.org/doc/grammar.html.  It is called from write_fst, and writes to
    the stdout some extra arcs in the lexicon FST that relate to nonterminal
    symbols.

    See the section "Special symbols in L.fst,
    kaldi-asr.org/doc/grammar.html#grammar_special_l.
       start_state: the start-state of L.fst.
       sil_state:  the state of high out-degree in L.fst where words leave
                   when preceded by optional silence
       non_sil_state:   the state of high out-degree in L.fst where words leave
                   when not preceded by optional silence
       next_state: the number from which this function can start allocating its
                  own states.  the updated value of next_state will be returned.
       sil_phone:  the optional-silence phone (a string, e.g 'sil')
       nonterminals: the user-defined nonterminal symbols as a list of
          strings, e.g. ['#nonterm:contact_list', ... ].
       left_context_phones: a list of phones that may appear as left-context,
          e.g. ['a', 'ah', ... '#nonterm_bos'].
    """
    shared_state = next_state
    next_state += 1
    final_state = next_state
    next_state += 1

    print("{src}\t{dest}\t{phone}\t{word}\t{cost}".format(
        src=start_state, dest=shared_state,
        phone='#nonterm_begin', word='#nonterm_begin',
        cost=0.0))

    for nonterminal in nonterminals:
        # What we are doing here could be viewed as a little lazy, by going to
        # 'shared_state' instead of a state specific to nonsilence vs. silence
        # left-context vs. unknown (for #nonterm_begin).  If we made them
        # separate we could improve (by half) the correctness of how it
        # interacts with sil-probs in the hard-to-handle case where
        # word-position-dependent phones are not used and some words end
        # in the optional-silence phone.
        for src in [sil_state, non_sil_state]:
            print("{src}\t{dest}\t{phone}\t{word}\t{cost}".format(
                src=src, dest=shared_state,
                phone=nonterminal, word=nonterminal,
                cost=0.0))

    # this_cost equals log(len(left_context_phones)) but the expression below
    # better captures the meaning.  Applying this cost to arcs keeps the FST
    # stochatic (sum-to-one, like an HMM), so that if we do weight pushing
    # things won't get weird.  In the grammar-FST code when we splice things
    # together we will cancel out this cost, see the function CombineArcs().
    this_cost = -math.log(1.0 / len(left_context_phones))

    for left_context_phone in left_context_phones:
        # The following line is part of how we get this to interact correctly with
        # the silence probabilities: if the left-context phone was the silence
        # phone, it goes to sil_state, else nonsil_state.  This won't always
        # do the right thing if you have a system without word-position-dependent
        # phones (--position-dependent-phones false to prepare_lang.sh) and
        # you have words that end in the optional-silence phone.
        dest = (sil_state if left_context_phone == sil_phone else non_sil_state)

        print("{src}\t{dest}\t{phone}\t{word}\t{cost}".format(
            src=shared_state, dest=dest,
            phone=left_context_phone, word='<eps>', cost=this_cost))

    # arc from sil_state and non_sil_state to a final-state with #nonterm_end as
    # ilabel and olabel.  The costs on these arcs are zero because if you take
    # that arc, you are not really terminating the sequence, you are just
    # skipping to sil_state or non_sil_state in the FST one level up.  It
    # takes the correct path because of the code around 'dest = ...' a few
    # lines above this, after reaching 'shared_state' because it saw the
    # user-defined nonterminal.
    for src in [sil_state, non_sil_state]:
        print("{src}\t{dest}\t{phone}\t{word}\t{cost}".format(
            src=src, dest=final_state,
            phone='#nonterm_end', word='#nonterm_end', cost=0.0))
    print("{state}\t{final_cost}".format(
        state=final_state, final_cost=0.0))
    return next_state

def write_fst(lexicon, silprobs, sil_phone, sil_disambig,
              nonterminals = None, left_context_phones = None):
    """Writes the text format of L.fst (or L_disambig.fst)  to the standard output.
     'lexicon' is a list of 5-tuples
     (word, pronprob, wordsilprob, silwordcorrection, nonsilwordcorrection, pron)
         as returned by read_lexiconp().
     'silprobs' is a 4-tuple of probabilities as returned by read_silprobs().
     'sil_phone' is the silence phone, e.g. "SIL".
     'sil_disambig' is either '<eps>', or the silence disambiguation symbol, e.g. "#5".
     'nonterminals', which relates to grammar decoding (see kaldi-asr.org/doc/grammar.html),
        is either None, or the user-defined nonterminal symbols as a list of
        strings, e.g. ['#nonterm:contact_list', ... ].
     'left_context_phones', which also relates to grammar decoding, and must be
        supplied if 'nonterminals' is supplied is either None or a list of
        phones that may appear as left-context, e.g. ['a', 'ah', ... '#nonterm_bos'].
    """
    silbeginprob, silendcorrection, nonsilendcorrection, siloverallprob = silprobs
    initial_sil_cost = -math.log(silbeginprob)
    initial_non_sil_cost = -math.log(1.0 - silbeginprob);
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
        phone=sil_disambig, word='<eps>', cost=initial_non_sil_cost))
    print('{src}\t{dest}\t{phone}\t{word}\t{cost}'.format(
        src=start_state, dest=sil_state,
        phone=sil_phone, word='<eps>', cost=initial_sil_cost))

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

        # add states and arcs for all but the first phone.
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

    if nonterminals is not None:
        next_state = write_nonterminal_arcs(
            start_state, sil_state, non_sil_state,
            next_state, sil_phone,
            nonterminals, left_context_phones)

    print('{src}\t{cost}'.format(src=sil_state, cost=sil_end_correction_cost))
    print('{src}\t{cost}'.format(src=non_sil_state, cost=non_sil_end_correction_cost))

def read_nonterminals(filename):
    """Reads the user-defined nonterminal symbols in 'filename', checks that
       it has the expected format and has no duplicates, and returns the nonterminal
       symbols as a list of strings, e.g.
       ['#nonterm:contact_list', '#nonterm:phone_number', ... ]. """
    ans = [line.strip(" \t\r\n") for line in open(filename, 'r', encoding='latin-1')]
    if len(ans) == 0:
        raise RuntimeError("The file {0} contains no nonterminals symbols.".format(filename))
    for nonterm in ans:
        if nonterm[:9] != '#nonterm:':
            raise RuntimeError("In file '{0}', expected nonterminal symbols to start with '#nonterm:', found '{1}'"
                               .format(filename, nonterm))
    if len(set(ans)) != len(ans):
        raise RuntimeError("Duplicate nonterminal symbols are present in file {0}".format(filename))
    return ans

def read_left_context_phones(filename):
    """Reads, checks, and returns a list of left-context phones, in text form, one
       per line.  Returns a list of strings, e.g. ['a', 'ah', ..., '#nonterm_bos' ]"""
    ans = [line.strip(" \t\r\n") for line in open(filename, 'r', encoding='latin-1')]
    if len(ans) == 0:
        raise RuntimeError("The file {0} contains no left-context phones.".format(filename))
    for s in ans:
        if len(s.split()) != 1:
            raise RuntimeError("The file {0} contains an invalid line '{1}'".format(filename, s)   )

    if len(set(ans)) != len(ans):
        raise RuntimeError("Duplicate nonterminal symbols are present in file {0}".format(filename))
    return ans


def main():
    args = get_args()
    silprobs = read_silprobs(args.silprobs)
    lexicon = read_lexiconp(args.lexiconp)


    if args.nonterminals is None:
        nonterminals, left_context_phones = None, None
    else:
        if args.left_context_phones is None:
            print("{0}: if --nonterminals is specified, --left-context-phones must also "
                  "be specified".format(sys.argv[0]))
            sys.exit(1)
        nonterminals = read_nonterminals(args.nonterminals)
        left_context_phones = read_left_context_phones(args.left_context_phones)

    write_fst(lexicon, silprobs, args.sil_phone, args.sil_disambig,
              nonterminals, left_context_phones)


if __name__ == '__main__':
      main()
