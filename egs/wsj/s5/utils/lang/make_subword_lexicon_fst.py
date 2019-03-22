#!/usr/bin/env python3

# 2019 Dongji Gao

from collections import namedtuple
from make_lexicon_fst import read_lexiconp
import argparse
import math
import sys

# see get_args() below for usage mesage
def get_args():
    parser = argparse.ArgumentParser(description="""This script creates a 
        position depent subword lexicon FST from a origin lexicon""")
    parser.add_argument('--sil-phone', type=str, help="""Text form of
        optional-silence phone.""")
    parser.add_argument('--sil-prob', type=float, default=0.0, help="""Probability
        of silence between words (including the beginning and end of word sequence).
        Must be in range [0.0, 1.0]""")
    parser.add_argument('--sil-disambig', type=str, help="""Disambiguation symbol
        to disambiguate silence, e.g. #5.""")
    parser.add_argument('--position-dependent', action="store_true", help="""Whether the input lexicon
        is position-dependent.""")
    parser.add_argument("--separator", type=str, default="@@", help="""Separator indicates the
        position of a subword in word.""")
    parser.add_argument('lexiconp', type=str, help="""Filename of lexicon with
        pronunciation probabilities (normally lexiconp.txt), with lines of the
        form 'word prob p1 p2...', e.g. 'a, 1.0 ay'""")
    args = parser.parse_args()
    return args

def contain_disambig_symbol(phones):
    """Disambig symbol is at the end of phones in the form of #1, #2..."""
    return True if phones[-1][0] == "#" else False

def print_arc(src, dest, phone, word, cost):
    print('{}\t{}\t{}\t{}\t{}'.format(src, dest, phone, word, cost))

def is_end(word, separator):
    return (not word.endswith(separator))

def get_suffix(phone):
    return phone[-2:]

def get_position(phones, constant_variable):
    phones = phones[:-1] if contain_disambig_symbol(phones) else phones

    if len(phones) == 1:
        suffix = get_suffix(phones[0])
        if suffix == "_B":
            return constant_variable.begin
        elif suffix == "_I":
            return constant_variable.middle
        elif suffix == "_E":
            return constant_variable.end
        else:
            return constant_variable.word
    else:
        if get_suffix(phones[0]) == "_B":
            if get_suffix(phones[-1]) == "_I":
                return constant_variable.begin
            elif get_suffix(phones[-1]) == "_E":
                return constant_variable.word
        elif get_suffix(phones[0]) == "_I":
            if get_suffix(phones[-1]) == "_I":
                return constant_variable.middle
            elif get_suffix(phones[-1]) == "_E":
                return constant_variable.end

def write_fst_no_silence(lexicon, position_dependent, separator, constant_variable):
    """Writes the text format of L.fst to the standard output.  This version is for
    when --sil-prob=0.0, meaning there is no optional silence allowed.

     'lexicon' is a list of 3-tuples (subword, pron-prob, prons) as returned by
     'position_dependent', which is True is the lexicion is position-dependent.
     'separator' is the symbol we ues to indicate the postion of a subword in word.
    """
    def print_word_no_silence(word, pron_cost, phones, position, next_state):
        
        assert(len(phones)) > 0

        from_state = prefix_state
        to_state = prefix_state
        if position == constant_variable.begin:
            from_state = loop_state
        elif position == constant_variable.end:
            to_state = loop_state
        elif position == constant_variable.word:
            from_state = loop_state
            to_state = loop_state

        current_state = from_state
        for i in range(len(phones) - 1):
            word = (word if i == 0 else constant_variable.eps)
            cost = (pron_cost if i == 0 else 0.0)
            print_arc(current_state, next_state, phones[i], word, cost)
            current_state = next_state
            next_state += 1

        i = len(phones) - 1
        phone = (phones[i] if i >= 0 else constant_variable.eps)
        word = (word if i <= 0 else constant_variable.eps)
        cost = (pron_cost if i <= 0 else 0.0)
        print_arc(current_state, to_state, phone, word, cost)

        return next_state

    loop_state = 0
    prefix_state = 1
    next_state = 2

    for (word, pron_prob, phones) in lexicon:
        cost = -math.log(pron_prob)

        if is_end(word,separator):
            if position_dependent:
                position = get_position(phones, constant_variable)
                next_state = print_word_no_silence(word, cost, phones, position, next_state)
            else:
                next_state = print_word_no_silence(word, cost, phones, constant_variable.end, next_state)
                next_state = print_word_no_silence(word, cost, phones, constant_variable.word, next_state)
        else:
            # the word end with separator which should be followed by subword
            if position_dependent:
                position = get_position(phones, constant_variable) 
                next_state = print_word_no_silence(word, cost, phones, position, next_state)
            else:
                next_state = print_word_no_silence(word, cost, phones, constant_variable.begin, next_state)
                next_state = print_word_no_silence(word, cost, phones, constant_variable.middle, next_state)

    print("{state}\t{final_cost}".format(state=loop_state, final_cost=0.0))

def write_fst_with_silence(lexicon, sil_phone, sil_prob, sil_disambig, 
    position_dependent, separator, constant_variable):
    """Writes the text format of L.fst to the standard output.  This version is for
       when --sil-prob != 0.0, meaning there is optional silence
     'lexicon' is a list of 3-tuples (subword, pron-prob, prons)
         as returned by read_lexiconp().
     'sil_prob', which is expected to be strictly between 0.0 and 1.0, is the
         probability of silence
     'sil_phone' is the silence phone, e.g. "SIL".
     'sil_disambig' is either None, or the silence disambiguation symbol, e.g. "#5".
     'position_dependent', which is True is the lexicion is position-dependent.
     'separator' is the symbol we ues to indicate the postion of a subword in word.
    """
    def print_word_with_silence(word, pron_cost, phones, position, next_state):
    
        assert(len(phones)) > 0
    
        from_state = prefix_state
        to_state = prefix_state
        if position == constant_variable.begin:
            from_state = loop_state
        elif position == constant_variable.end: 
            to_state = constant_variable.sil_nonsil
        elif position == constant_variable.word:
            from_state = loop_state
            to_state = constant_variable.sil_nonsil
    
        current_state = from_state
        for i in range(len(phones) - 1):
            word = (word if i == 0 else constant_variable.eps)
            cost = (pron_cost if i == 0 else 0.0)
            print_arc(current_state, next_state, phones[i], word, cost)
            current_state = next_state
            next_state += 1
    
        i = len(phones) - 1
        phone = (phones[i] if i >= 0 else constant_variable.eps)
        word = (word if i <= 0 else constant_variable.eps)
        cost = (pron_cost if i <= 0 else 0.0)
        if to_state != constant_variable.sil_nonsil:
            print_arc(current_state, to_state, phone, word, cost)
        else:
            p_cost = cost
            cost = no_sil_cost + p_cost
            print_arc(current_state, loop_state, phone, word, cost)
            cost = sil_cost + p_cost
            print_arc(current_state, sil_state, phone, word, cost)
    
        return next_state

    sil_cost = -math.log(sil_prob)
    no_sil_cost = -math.log(1 - sil_prob)

    start_state = 0
    loop_state = 1
    sil_state = 2
    prefix_state = 3
    next_state = 4

    print_arc(start_state, loop_state, constant_variable.eps, constant_variable.eps, no_sil_cost)
    print_arc(start_state, sil_state, constant_variable.eps, constant_variable.eps, sil_cost)
    if sil_disambig is None:
        print_arc(sil_state, loop_state, sil_phone, constant_variable.eps, 0.0)
    else:
        disambig_state = next_state
        next_state += 1
        print_arc(sil_state, disambig_state, sil_phone, constant_variable.eps, 0.0)
        print_arc(disambig_state, loop_state, sil_disambig, constant_variable.eps, cost=0.0)

    for (word, pron_prob, phones) in lexicon:
        cost = -math.log(pron_prob)

        if is_end(word, separator):
            if position_dependent:
                position = get_position(phones, constant_variable)
                next_state = print_word_with_silence(word, cost, phones, position, next_state)
            else:
                next_state = print_word_with_silence(word, cost, phones, constant_variable.end, next_state)
                next_state = print_word_with_silence(word, cost, phones, constant_variable.word, next_state)
        else:
            # the word end with separator which should be followed by subword
            if position_dependent:
                position = get_position(phones, constant_variable)
                next_state = print_word_with_silence(word, cost, phones, position, next_state)
            else:
                next_state = print_word_with_silence(word, cost, phones, constant_variable.begin, next_state)
                next_state = print_word_with_silence(word, cost, phones, constant_variable.middle, next_state)

    print("{state}\t{final_cost}".format(state=loop_state, final_cost=0.0))

def main():
    variable = namedtuple("variable", "eps begin middle end word sil_nonsil")
    constant_variable = variable(eps="<eps>", begin="begin", middle="middle", 
        end="end", word="word", sil_nonsil="sil_nonsil")

    args = get_args()
    if args.sil_prob < 0.0 or args.sil_prob >= 1.0:
        print("{}: invalid value specified --sil-prob={}".format(
              sys.argv[0], args.sil_prob), file=sys.stderr)
    lexicon = read_lexiconp(args.lexiconp)
    if args.sil_prob == 0.0:
        write_fst_no_silence(lexicon, args.position_dependent, args.separator,constant_variable)
    else:
        write_fst_with_silence(lexicon, args.sil_phone, args.sil_prob, 
            args.sil_disambig, args.position_dependent, args.separator, constant_variable)

if __name__ == "__main__":
    main()
