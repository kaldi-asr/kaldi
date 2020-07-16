#!/usr/bin/env python3

# Copyright   2018  Johns Hopkins University (author: Daniel Povey)
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
       the stdout.""")

    parser.add_argument('--sil-phone', dest='sil_phone', type=str,
                        help="""Text form of optional-silence phone, e.g. 'SIL'.  See also
                        the --silprob option.""")
    parser.add_argument('--sil-prob', dest='sil_prob', type=float, default=0.0,
                        help="""Probability of silence between words (including at the
                        beginning and end of word sequences).  Must be in the range [0.0, 1.0].
                        This refers to the optional silence inserted by the lexicon; see
                        the --silphone option.""")
    parser.add_argument('--sil-disambig', dest='sil_disambig', type=str,
                        help="""Disambiguation symbol to disambiguate silence, e.g. #5.
                        Will only be supplied if you are creating the version of L.fst
                        with disambiguation symbols, intended for use with cyclic G.fst.
                        This symbol was introduced to fix a rather obscure source of
                        nondeterminism of CLG.fst, that has to do with reordering of
                        disambiguation symbols and phone symbols.""")
    parser.add_argument('--left-context-phones', dest='left_context_phones', type=str,
                        help="""Only relevant if --nonterminals is also supplied; this relates
                        to grammar decoding (see http://kaldi-asr.org/doc/grammar.html or
                        src/doc/grammar.dox).  Format is a list of left-context phones,
                        in text form, one per line.  E.g. data/lang/phones/left_context_phones.txt""")
    parser.add_argument('--nonterminals', type=str,
                        help="""If supplied, --left-context-phones must also be supplied.
                        List of user-defined nonterminal symbols such as #nonterm:contact_list,
                        one per line.  E.g. data/local/dict/nonterminals.txt.""")
    parser.add_argument('lexiconp', type=str,
                        help="""Filename of lexicon with pronunciation probabilities
                        (normally lexiconp.txt), with lines of the form 'word prob p1 p2...',
                        e.g. 'a   1.0    ay'""")
    args = parser.parse_args()
    return args


def read_lexiconp(filename):
    """Reads the lexiconp.txt file in 'filename', with lines like 'word pron p1 p2 ...'.
    Returns a list of tuples (word, pron_prob, pron), where 'word' is a string,
   'pron_prob', a float, is the pronunciation probability (which must be >0.0
    and would normally be <=1.0),  and 'pron' is a list of strings representing phones.
    An element in the returned list might be ('hello', 1.0, ['h', 'eh', 'l', 'ow']).
    """

    ans = []
    found_empty_prons = False
    found_large_pronprobs = False
    # See the comment near the top of this file, RE why we use latin-1.
    with open(filename, 'r', encoding='latin-1') as f:
        whitespace = re.compile("[ \t]+")
        for line in f:
            a = whitespace.split(line.strip(" \t\r\n"))
            if len(a) < 2:
                print("{0}: error: found bad line '{1}' in lexicon file {2} ".format(
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
            except:
                print("{0}: error: found bad line '{1}' in lexicon file {2}, 2nd field "
                      "should be pron-prob".format(sys.argv[0], line.strip(" \t\r\n"), filename),
                      file=sys.stderr)
                sys.exit(1)
            prons = a[2:]
            if pron_prob <= 0.0:
                print("{0}: error: invalid pron-prob in line '{1}' of lexicon file {1} ".format(
                    sys.argv[0], line.strip(" \t\r\n"), filename), file=sys.stderr)
                sys.exit(1)
            if len(prons) == 0:
                found_empty_prons = True
            ans.append( (word, pron_prob, prons) )
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


def write_nonterminal_arcs(start_state, loop_state, next_state,
                           nonterminals, left_context_phones):
    """This function relates to the grammar-decoding setup, see
    kaldi-asr.org/doc/grammar.html.  It is called from write_fst_no_silence
    and write_fst_silence, and writes to the stdout some extra arcs
    in the lexicon FST that relate to nonterminal symbols.
    See the section "Special symbols in L.fst,
    kaldi-asr.org/doc/grammar.html#grammar_special_l.
       start_state: the start-state of L.fst.
       loop_state:  the state of high out-degree in L.fst where words leave
                  and enter.
       next_state: the number from which this function can start allocating its
                  own states.  the updated value of next_state will be returned.
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
        print("{src}\t{dest}\t{phone}\t{word}\t{cost}".format(
            src=loop_state, dest=shared_state,
            phone=nonterminal, word=nonterminal,
            cost=0.0))
    # this_cost equals log(len(left_context_phones)) but the expression below
    # better captures the meaning.  Applying this cost to arcs keeps the FST
    # stochatic (sum-to-one, like an HMM), so that if we do weight pushing
    # things won't get weird.  In the grammar-FST code when we splice things
    # together we will cancel out this cost, see the function CombineArcs().
    this_cost = -math.log(1.0 / len(left_context_phones))

    for left_context_phone in left_context_phones:
        print("{src}\t{dest}\t{phone}\t{word}\t{cost}".format(
            src=shared_state, dest=loop_state,
            phone=left_context_phone, word='<eps>', cost=this_cost))
    # arc from loop-state to a final-state with #nonterm_end as ilabel and olabel
    print("{src}\t{dest}\t{phone}\t{word}\t{cost}".format(
        src=loop_state, dest=final_state,
        phone='#nonterm_end', word='#nonterm_end', cost=0.0))
    print("{state}\t{final_cost}".format(
        state=final_state, final_cost=0.0))
    return next_state



def write_fst_no_silence(lexicon, nonterminals=None, left_context_phones=None):
    """Writes the text format of L.fst to the standard output.  This version is for
    when --sil-prob=0.0, meaning there is no optional silence allowed.

      'lexicon' is a list of 3-tuples (word, pron-prob, prons) as returned by
        read_lexiconp().
     'nonterminals', which relates to grammar decoding (see kaldi-asr.org/doc/grammar.html),
        is either None, or the user-defined nonterminal symbols as a list of
        strings, e.g. ['#nonterm:contact_list', ... ].
     'left_context_phones', which also relates to grammar decoding, and must be
        supplied if 'nonterminals' is supplied is either None or a list of
        phones that may appear as left-context, e.g. ['a', 'ah', ... '#nonterm_bos'].
    """

    loop_state = 0
    next_state = 1  # the next un-allocated state, will be incremented as we go.
    for (word, pronprob, pron) in lexicon:
        cost = -math.log(pronprob)
        cur_state = loop_state
        for i in range(len(pron) - 1):
            print("{src}\t{dest}\t{phone}\t{word}\t{cost}".format(
                src=cur_state,
                dest=next_state,
                phone=pron[i],
                word=(word if i == 0 else '<eps>'),
                cost=(cost if i == 0 else 0.0)))
            cur_state = next_state
            next_state += 1

        i = len(pron) - 1  # note: i == -1 if pron is empty.
        print("{src}\t{dest}\t{phone}\t{word}\t{cost}".format(
            src=cur_state,
            dest=loop_state,
            phone=(pron[i] if i >= 0 else '<eps>'),
            word=(word if i <= 0 else '<eps>'),
            cost=(cost if i <= 0 else 0.0)))

    if nonterminals is not None:
        next_state = write_nonterminal_arcs(
            loop_state, loop_state, next_state,
            nonterminals, left_context_phones)

    print("{state}\t{final_cost}".format(
        state=loop_state,
        final_cost=0.0))


def write_fst_with_silence(lexicon, sil_prob, sil_phone, sil_disambig,
                           nonterminals=None, left_context_phones=None):
    """Writes the text format of L.fst to the standard output.  This version is for
       when --sil-prob != 0.0, meaning there is optional silence
     'lexicon' is a list of 3-tuples (word, pron-prob, prons)
         as returned by read_lexiconp().
     'sil_prob', which is expected to be strictly between 0.. and 1.0, is the
         probability of silence
     'sil_phone' is the silence phone, e.g. "SIL".
     'sil_disambig' is either None, or the silence disambiguation symbol, e.g. "#5".
     'nonterminals', which relates to grammar decoding (see kaldi-asr.org/doc/grammar.html),
        is either None, or the user-defined nonterminal symbols as a list of
        strings, e.g. ['#nonterm:contact_list', ... ].
     'left_context_phones', which also relates to grammar decoding, and must be
        supplied if 'nonterminals' is supplied is either None or a list of
        phones that may appear as left-context, e.g. ['a', 'ah', ... '#nonterm_bos'].
    """

    assert sil_prob > 0.0 and sil_prob < 1.0
    sil_cost = -math.log(sil_prob)
    no_sil_cost = -math.log(1.0 - sil_prob);

    start_state = 0
    loop_state = 1  # words enter and leave from here
    sil_state = 2   # words terminate here when followed by silence; this state
                    # has a silence transition to loop_state.
    next_state = 3  # the next un-allocated state, will be incremented as we go.


    print('{src}\t{dest}\t{phone}\t{word}\t{cost}'.format(
        src=start_state, dest=loop_state,
        phone='<eps>', word='<eps>', cost=no_sil_cost))
    print('{src}\t{dest}\t{phone}\t{word}\t{cost}'.format(
        src=start_state, dest=sil_state,
        phone='<eps>', word='<eps>', cost=sil_cost))
    if sil_disambig is None:
        print('{src}\t{dest}\t{phone}\t{word}\t{cost}'.format(
            src=sil_state, dest=loop_state,
            phone=sil_phone, word='<eps>', cost=0.0))
    else:
        sil_disambig_state = next_state
        next_state += 1
        print('{src}\t{dest}\t{phone}\t{word}\t{cost}'.format(
            src=sil_state, dest=sil_disambig_state,
            phone=sil_phone, word='<eps>', cost=0.0))
        print('{src}\t{dest}\t{phone}\t{word}\t{cost}'.format(
            src=sil_disambig_state, dest=loop_state,
            phone=sil_disambig, word='<eps>', cost=0.0))


    for (word, pronprob, pron) in lexicon:
        pron_cost = -math.log(pronprob)
        cur_state = loop_state
        for i in range(len(pron) - 1):
            print("{src}\t{dest}\t{phone}\t{word}\t{cost}".format(
                src=cur_state, dest=next_state,
                phone=pron[i],
                word=(word if i == 0 else '<eps>'),
                cost=(pron_cost if i == 0 else 0.0)))
            cur_state = next_state
            next_state += 1

        i = len(pron) - 1  # note: i == -1 if pron is empty.
        print("{src}\t{dest}\t{phone}\t{word}\t{cost}".format(
            src=cur_state,
            dest=loop_state,
            phone=(pron[i] if i >= 0 else '<eps>'),
            word=(word if i <= 0 else '<eps>'),
            cost=no_sil_cost + (pron_cost if i <= 0 else 0.0)))
        print("{src}\t{dest}\t{phone}\t{word}\t{cost}".format(
            src=cur_state,
            dest=sil_state,
            phone=(pron[i] if i >= 0 else '<eps>'),
            word=(word if i <= 0 else '<eps>'),
            cost=sil_cost + (pron_cost if i <= 0 else 0.0)))

    if nonterminals is not None:
        next_state = write_nonterminal_arcs(
            start_state, loop_state, next_state,
            nonterminals, left_context_phones)

    print("{state}\t{final_cost}".format(
        state=loop_state,
        final_cost=0.0))




def write_words_txt(orig_lines, highest_numbered_symbol, nonterminals, filename):
    """Writes updated words.txt to 'filename'.  'orig_lines' is the original lines
       in the words.txt file as a list of strings (without the newlines);
       highest_numbered_symbol is the highest numbered symbol in the original
       words.txt; nonterminals is a list of strings like '#nonterm:foo'."""
    with open(filename, 'w', encoding='latin-1') as f:
        for l in orig_lines:
            print(l, file=f)
        cur_symbol = highest_numbered_symbol + 1
        for n in [ '#nonterm_begin', '#nonterm_end' ] + nonterminals:
            print("{0} {1}".format(n, cur_symbol), file=f)
            cur_symbol = cur_symbol + 1


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
    whitespace = re.compile("[ \t]+")
    for s in ans:
        if len(whitespace.split(s)) != 1:
            raise RuntimeError("The file {0} contains an invalid line '{1}'".format(filename, s)   )

    if len(set(ans)) != len(ans):
        raise RuntimeError("Duplicate nonterminal symbols are present in file {0}".format(filename))
    return ans


def is_token(s):
    """Returns true if s is a string and is space-free."""
    if not isinstance(s, str):
        return False
    whitespace = re.compile("[ \t\r\n]+")
    split_str = whitespace.split(s);
    return len(split_str) == 1 and s == split_str[0]


def main():
    args = get_args()

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

    if args.sil_prob == 0.0:
          write_fst_no_silence(lexicon,
                               nonterminals=nonterminals,
                               left_context_phones=left_context_phones)
    else:
        # Do some checking that the options make sense.
        if args.sil_prob < 0.0 or args.sil_prob >= 1.0:
            print("{0}: invalid value specified --sil-prob={1}".format(
                sys.argv[0], args.sil_prob), file=sys.stderr)
            sys.exit(1)

        if not is_token(args.sil_phone):
            print("{0}: you specified --sil-prob={1} but --sil-phone is set "
                  "to '{2}'".format(sys.argv[0], args.sil_prob, args.sil_phone),
                  file=sys.stderr)
            sys.exit(1)
        if args.sil_disambig is not None and not is_token(args.sil_disambig):
            print("{0}: invalid value --sil-disambig='{1}' was specified."
                  "".format(sys.argv[0], args.sil_disambig), file=sys.stderr)
            sys.exit(1)
        write_fst_with_silence(lexicon, args.sil_prob, args.sil_phone,
                               args.sil_disambig,
                               nonterminals=nonterminals,
                               left_context_phones=left_context_phones)



#    (lines, highest_symbol) = read_words_txt(args.input_words_txt)
#    nonterminals = read_nonterminals(args.nonterminal_symbols_list)
#    write_words_txt(lines, highest_symbol, nonterminals, args.output_words_txt)


if __name__ == '__main__':
      main()
