#!/usr/bin/env python3

# 2019 Dongji Gao
# Apache 2.0.

from make_lexicon_fst import read_lexiconp
import argparse
import math
import sys

# see get_args() below for usage mesage
def get_args():
    parser = argparse.ArgumentParser(description="""This script creates the
        text form of a subword lexicon FST to be compiled by fstcompile using
        the appropriate symbol tables (phones.txt and words.txt). It will mostly
        be invoked indirectly via utils/prepare_lang_subword.sh. The output
        goes to the stdout. This script is the subword version of make_lexicon_fst.py.
        It only allows optional silence to appear after end-subword or singleton-subword,
        (i.e., subwords without separator). In this version we do not support
        pronunciation probability. (i.e., pron-prob = 1.0)""")

    parser.add_argument('--sil-phone', type=str, help="""Text form of
        optional-silence phone, e.g. 'SIL'. See also the --sil-prob option.""")
    parser.add_argument('--sil-prob', type=float, default=0.0, help="""Probability
        of silence between words (including the beginning and end of word sequence).
        Must be in range [0.0, 1.0). This refer to the optional silence inserted by
        the lexicon; see the --sil-phone option.""")
    parser.add_argument('--sil-disambig', type=str, help="""Disambiguation symbol
        to disambiguate silence, e.g. #5. Will only be supplied if you are creating 
        the version of L.fst with disambiguation symbols, intended for use with cyclic 
        G.fst. This symbol was introduced to fix a rather obscure source of nondeterminism 
        of CLG.fst, that has to do with reordering of disambiguation symbols and phone symbols.""")
    parser.add_argument('--position-dependent', action="store_true", help="""Whether 
        the input lexicon is position-dependent.""")
    parser.add_argument("--separator", type=str, default="@@", help="""Separator
        indicates the position of a subword in a word.
        Subword followed by separator can only appear at the beginning or middle of a word.
        Subword without separator can only appear at the end of a word or is a word itself.
        E.g. "international -> inter@@ nation@@ al";
             "nation        -> nation"
    The separator should match the separator used in the input lexicon.""")
    parser.add_argument('lexiconp', type=str, help="""Filename of lexicon with
        pronunciation probabilities (normally lexiconp.txt), with lines of the
        form 'subword prob p1 p2...', e.g. 'a, 1.0 ay'""")
    args = parser.parse_args()
    return args

def contain_disambig_symbol(phones):
    """Return true if the phone sequence contains disambiguation symbol.
    Return false otherwise. Disambiguation symbol is at the end of phones 
    in the form of #1, #2... There is at most one disambiguation 
    symbol for each phone sequence"""
    return True if phones[-1].startswith("#") else False

def print_arc(src, dest, phone, word, cost):
    print('{}\t{}\t{}\t{}\t{}'.format(src, dest, phone, word, cost))

def is_end(word, separator):
    """Return true if the subword can appear at the end of a word (i.e., the subword
    does not end with separator). Return false otherwise."""
    return not word.endswith(separator)

def get_suffix(phone):
    """Return the suffix of a phone. The suffix is in the form of '_B', '_I'..."""
    if len(phone) < 3:
        print("{}: invalid phone {} (please check if the phone is position-dependent)".format(
              sys.argv[0], phone), file=sys.stderr)
        sys.exit(1)
    return phone[-2:]

def write_fst_no_silence(lexicon, position_dependent, separator):
    """Writes the text format of L.fst to the standard output.  This version is for
    when --sil-prob=0.0, meaning there is no optional silence allowed.
    loop_state here is the start and final state of the fst. It goes to word_start_state
    via epsilon transition.
    In position-independent case, there is no difference between beginning word and 
    middle word. So all subwords with separator would leave from and enter word_start_state.
    All subword without separator would leave from word_start_state and enter loop_state.
    This guarantees that optional silence can only follow a word-end subword.

    In position-dependent case, there are 4 types of position-dependent subword:
    1) Beginning subword. The first phone suffix should be "_B" and other suffixes should be "_I"s:
        nation@@ 1.0 n_B ey_I sh_I ih_I n_I
        n@@      1.0 n_B
    2) Middle subword. All phone suffixes should be "_I"s:
        nation@@ 1.0 n_I ey_I sh_I ih_I n_I
    3) End subword. The last phone suffix should be "_E" and other suffixes be should "_I"s:
        nation   1.0 n_I ey_I sh_I ih_I n_E
        n        1.0 n_E
    4) Singleton subword (i.e., the subword is word it self).
       The first phone suffix should be "_B" and the last suffix should be "_E".
       All other suffix should be "_I"s. If there is only one phone, its suffix should be "_S":
        nation   1.0 n_B ey_I sh_I ih_I n_E
        n        1.0 n_S

    So we need an extra word_internal_state. The beginning word 
    would leave from word_start_state and enter word_internal_state and middle word
    would leave from and enter word_internal_state. The rest part is same.

      'lexicon' is a list of 3-tuples (subword, pron-prob, prons) as returned by
      'position_dependent', which is true is the lexicon is position-dependent.
      'separator' is a symbol which indicates the position of a subword in word.
    """
    # regular setting
    loop_state = 0
    word_start_state = 1
    next_state = 2

    print_arc(loop_state, word_start_state, "<eps>", "<eps>", 0.0)

    # optional setting for word_internal_state
    if position_dependent:
        word_internal_state = next_state
        next_state += 1

    for (word, pron_prob, phones) in lexicon:
        pron_cost = 0.0                # do not support pron_prob
        phones_len = len(phones)

        # set start and end state for different cases
        if position_dependent:
            first_phone_suffix = get_suffix(phones[0])
            last_phone = phones[-2] if contain_disambig_symbol(phones) else phones[-1]
            last_phone_suffix = get_suffix(last_phone)

            # singleton word
            if first_phone_suffix == "_S":
                current_state = word_start_state
                end_state = loop_state
            # set the current_state
            elif first_phone_suffix == "_B":
                current_state = word_start_state
            elif first_phone_suffix == "_I" or first_phone_suffix == "_E":
                current_state = word_internal_state
            # then set the end_state
            if last_phone_suffix == "_B" or last_phone_suffix == "_I":
                end_state = word_internal_state
            elif last_phone_suffix == "_E":
                end_state = loop_state
        else:
            current_state = word_start_state
            end_state = loop_state if is_end(word, separator) else word_start_state

        # print arcs (except the last one) for the subword
        for i in range(phones_len - 1):
            word = word if i == 0 else "<eps>"
            cost = pron_cost if i == 0 else 0.0
            print_arc(current_state, next_state, phones[i], word, cost)
            current_state = next_state
            next_state += 1

        # print the last arc
        i = phones_len - 1
        phone = phones[i] if i >=0 else "<eps>"
        word = word if i <= 0 else "<eps>"
        cost = pron_cost if i <= 0 else 0.0
        print_arc(current_state, end_state, phone, word, cost)

    # set the final state
    print("{state}\t{final_cost}".format(state=loop_state, final_cost=0.0))

def write_fst_with_silence(lexicon, sil_phone, sil_prob, sil_disambig, position_dependent, separator):
    """Writes the text format of L.fst to the standard output.  This version is for
    when --sil-prob=0.0, meaning there is no optional silence allowed.
    loop_state here is the start and final state of the fst. It goes to word_start_state
    via epsilon transition.

    In position-independent case, there is no difference between beginning word and 
    middle word. So all subwords with separator would leave from and enter word_start_state.
    All subword without separator would leave from word_start_state and enter sil_state.
    This guarantees that optional silence can only follow a word-end subword and such subwords
    must appear at the end of the whole subword sequence.

    In position-dependent case, there are 4 types of position-dependent subword:
    1) Beginning subword. The first phone suffix should be "_B" and other suffixes should be "_I"s:
        nation@@ 1.0 n_B ey_I sh_I ih_I n_I
        n@@      1.0 n_B
    2) Middle subword. All phone suffixes should be "_I"s:
        nation@@ 1.0 n_I ey_I sh_I ih_I n_I
    3) End subword. The last phone suffix should be "_E" and other suffixes be should "_I"s:
        nation   1.0 n_I ey_I sh_I ih_I n_E
        n        1.0 n_E
    4) Singleton subword (i.e., the subword is word it self).
       The first phone suffix should be "_B" and the last suffix should be "_E".
       All other suffix should be "_I"s. If there is only one phone, its suffix should be "_S":
        nation   1.0 n_B ey_I sh_I ih_I n_E
        n        1.0 n_S

    So we need an extra word_internal_state. The beginning word 
    would leave from word_start_state and enter word_internal_state and middle word
    would leave from and enter word_internal_state. The rest part is same.

      'lexicon' is a list of 3-tuples (subword, pron-prob, prons)
         as returned by read_lexiconp().
      'sil_prob', which is expected to be strictly between 0.0 and 1.0, is the
         probability of silence
      'sil_phone' is the silence phone, e.g. "SIL".
      'sil_disambig' is either None, or the silence disambiguation symbol, e.g. "#5".
      'position_dependent', which is True is the lexicion is position-dependent.
      'separator' is the symbol we use to indicate the position of a subword in word.
    """

    sil_cost = -math.log(sil_prob)
    no_sil_cost = -math.log(1 - sil_prob)

    # regular setting
    start_state = 0
    loop_state = 1         # also the final state
    sil_state = 2          # words terminate here when followed by silence; this state
                           # has a licence transition to loop_state
    word_start_state = 3   # subword leave from here
    next_state = 4         # the next un-allocated state, will be incremented as we go

    print_arc(start_state, loop_state, "<eps>", "<eps>", no_sil_cost)
    print_arc(start_state, sil_state, "<eps>", "<eps>", sil_cost)
    print_arc(loop_state, word_start_state, "<eps>", "<eps>", 0.0)

    # optional setting for disambig_state
    if sil_disambig is None:
        print_arc(sil_state, loop_state, sil_phone, "<eps>", 0.0)
    else:
        disambig_state = next_state
        next_state += 1
        print_arc(sil_state, disambig_state, sil_phone, "<eps>", 0.0)
        print_arc(disambig_state, loop_state, sil_disambig, "<eps>", 0.0)

    # optional setting for word_internal_state
    if position_dependent:
        word_internal_state = next_state
        next_state += 1

    for (word, pron_prob, phones) in lexicon:
        pron_cost = 0.0           # do not support pron_prob
        phones_len = len(phones)
        
        # set start and end state for different cases
        if position_dependent:
            first_phone_suffix = get_suffix(phones[0])
            last_phone = phones[-2] if contain_disambig_symbol(phones) else phones[-1]
            last_phone_suffix = get_suffix(last_phone)

            # singleton subword
            if first_phone_suffix == "_S":
                current_state = word_start_state
                end_state_list = [loop_state, sil_state]
                end_cost_list = [no_sil_cost, sil_cost]
            # first set the current_state
            elif first_phone_suffix == "_B":
                current_state = word_start_state
            elif first_phone_suffix == "_I" or first_phone_suffix == "_E":
                current_state = word_internal_state
            # then set the end_state (end_state_list)
            if last_phone_suffix == "_B" or last_phone_suffix == "_I":
                end_state_list = [word_internal_state]
                end_cost_list = [0.0]
            elif last_phone_suffix == "_E":
                end_state_list = [loop_state, sil_state]
                end_cost_list = [no_sil_cost, sil_cost]
        else:
            current_state = word_start_state
            if is_end(word, separator):
                end_state_list = [loop_state, sil_state]
                end_cost_list = [no_sil_cost, sil_cost]
            else:
                end_state_list = [word_start_state]
                end_cost_list = [0.0]

        # print arcs (except the last one) for the subword
        for i in range(phones_len - 1):
            word = word if i == 0 else "<eps>"
            cost = pron_cost if i == 0 else 0.0
            print_arc(current_state, next_state, phones[i], word, cost)
            current_state = next_state
            next_state += 1

        # print the last arc
        i = phones_len - 1
        phone = phones[i] if i >= 0 else "<eps>"
        word = word if i <= 0 else "<eps>"
        cost = pron_cost if i <= 0 else 0.0
        for (end_state, end_cost) in zip(end_state_list, end_cost_list):
            print_arc(current_state, end_state, phone, word, cost + end_cost)

    # set the final state
    print("{state}\t{final_cost}".format(state=loop_state, final_cost=0.0))

def main():
    args = get_args()
    if args.sil_prob < 0.0 or args.sil_prob >= 1.0:
        print("{}: invalid value specified --sil-prob={}".format(
              sys.argv[0], args.sil_prob), file=sys.stderr)
        sys.exit(1)
    lexicon = read_lexiconp(args.lexiconp)
    if args.sil_prob == 0.0:
        write_fst_no_silence(lexicon, args.position_dependent, args.separator)
    else:
        write_fst_with_silence(lexicon, args.sil_phone, args.sil_prob, 
            args.sil_disambig, args.position_dependent, args.separator)

if __name__ == "__main__":
    main()
