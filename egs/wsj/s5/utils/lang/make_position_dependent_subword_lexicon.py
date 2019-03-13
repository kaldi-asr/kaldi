#!/usr/bin/env python3

# 2019 Dongji Gao

from make_lexicon_fst import read_lexiconp
import argparse
import math

def get_args():
    parser = argparse.ArgumentParser(description="""This script creates a
        opsition dependent subword lexicon from a subword lexicon.""")
    parser.add_argument("--separator", type=str, default="@@", help="""Separator
        indicate the positon of a subword in word.""")
    parser.add_argument("lexiconp", type=str, help="""Filename of subword lexicon
        with pronunciation probabilities, with lines of the form
        'subword prob p1 p2 ...'""")
    args = parser.parse_args()
    return args

def is_end(word, separator):
    return (not word.endswith(separator))

def write_position_dependent_lexicon(lexiconp, separator):
    for (word, prob, phones) in lexiconp:
        phones_length = len(phones)
        suffix_list = ["_I" for i in range(phones_length)]
        if not is_end(word, separator):
            # print middle
            phones_list = [phone + suffix for phone, suffix in zip (phones, suffix_list)]
            print("{} {} {}".format(word, prob, ' '.join(phones_list)))
            # print begin
            suffix_list[0] = "_B"
            phones_list = [phone + suffix for (phone, suffix) in zip (phones, suffix_list)]
            print("{} {} {}".format(word, prob, ' '.join(phones_list)))
        else:
            # print end 
            suffix_list[-1] = "_E"
            phones_list = [phone + suffix for (phone, suffix) in zip (phones, suffix_list)]
            print("{} {} {}".format(word, prob, ' '.join(phones_list)))
            #  print single
            if phones_length == 1:
                suffix_list[0] = "_S"
                phones_list = [phone + suffix for (phone, suffix) in zip (phones, suffix_list)]
                print("{} {} {}".format(word, prob, ' '.join(phones_list)))
            else:
                suffix_list[0] = "_B"
                phones_list = [phone + suffix for (phone, suffix) in zip (phones, suffix_list)]
                print("{} {} {}".format(word, prob, ' '.join(phones_list)))

def main():
    args = get_args()
    lexiconp = read_lexiconp(args.lexiconp)
    write_position_dependent_lexicon(lexiconp, args.separator)

if __name__ == "__main__":
    main()
