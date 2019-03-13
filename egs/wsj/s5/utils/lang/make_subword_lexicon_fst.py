#!/usr/bin/env python3

# 2019 Dongji Gao

from make_lexicon_fst import read_lexiconp
import argparse
import math


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

def write_fst_no_silence(lexicon):
    pass

def contain_disambig_symbol(phones):
    return True if phones[-1][0] == "#" else False


def write_fst_with_silence(lexicon, sil_phone, sil_prob, sil_disambig, position_dependent, separator):

    def print_arc(src, dest, phone, word, cost):
        print('{}\t{}\t{}\t{}\t{}'.format(src, dest, phone, word, cost))

    def is_end(word, separator):
        return (not word.endswith(separator))

    def get_suffix(phone):
        return phone[-2:]

    def get_position(phones):
        phones = phones[:-1] if contain_disambig_symbol(phones) else phones

        if len(phones) == 1:
            suffix = get_suffix(phones[0])
            if suffix == "_B":
                return "begin"
            elif suffix == "_I":
                return "middle"
            elif suffix == "_E":
                return "end"
            else:
                return "word"
        else:
            if get_suffix(phones[0]) == "_B":
                if get_suffix(phones[-1]) == "_I":
                    return "begin"
                elif get_suffix(phones[-1]) == "_E":
                    return "word"
            elif get_suffix(phones[0]) == "_I":
                if get_suffix(phones[-1]) == "_I":
                    return "middle"
                elif get_suffix(phones[-1]) == "_E":
                    return "end"

    def print_word(word, pron_cost, phones, position, next_state):
    
        assert(len(phones)) >= 0
    
        from_state = prefix_state
        to_state = prefix_state
        if position == "begin":
            from_state = loop_state
        elif position == "end": 
            to_state = "sil_nonsil"
        elif position == "word":
            from_state = loop_state
            to_state = "sil_nonsil"
    
        current_state = from_state
        for i in range(len(phones) - 1):
            word = (word if i == 0 else '<eps>')
            cost = (pron_cost if i == 0 else 0.0)
            print_arc(current_state, next_state, phones[i], word, cost)
            current_state = next_state
            next_state += 1
    
        i = len(phones) - 1
        phone = (phones[i] if i >= 0 else "<eps>")
        word = (word if i <= 0 else "<eps>")
        cost = (pron_cost if i <= 0 else 0.0)
        if to_state != "sil_nonsil":
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

    print_arc(start_state, loop_state, '<eps>', '<eps>', no_sil_cost)
    print_arc(start_state, sil_state, '<eps>', '<eps>', sil_cost)
    if sil_disambig is None:
        print_arc(sil_state, loop_state, sil_phone, '<eps>', 0.0)
    else:
        disambig_state = next_state
        next_state += 1
        print_arc(sil_state, disambig_state, sil_phone, '<eps>', 0.0)
        print_arc(disambig_state, loop_state, sil_disambig, '<eps>', cost=0.0)

    for (word, pron_prob, phones) in lexicon:
        cost = -math.log(pron_prob)

        if is_end(word, separator):
            if position_dependent:
                position = get_position(phones)
                next_state = print_word(word, cost, phones, position, next_state)
            else:
                next_state = print_word(word, cost, phones, "end", next_state)
                next_state = print_word(word, cost, phones, "word", next_state)
        else:
            # the word end with separator, meaning it should be followed by subword
            if position_dependent:
                position = get_position(phones)
                next_state = print_word(word, cost, phones, position, next_state)
            else:
                next_state = print_word(word, cost, phones, "begin", next_state)
                next_state = print_word(word, cost, phones, "middle", next_state)

    print("{state}\t{final_cost}".format(state=loop_state, final_cost=0.0))

def main():
    args = get_args()
    lexicon = read_lexiconp(args.lexiconp)
    if args.sil_prob == 0.0:
        write_fst_no_silence(lexicon)
    else:
        write_fst_with_silence(lexicon, args.sil_phone, args.sil_prob, args.sil_disambig, args.position_dependent, args.separator)

if __name__ == "__main__":
    main()
