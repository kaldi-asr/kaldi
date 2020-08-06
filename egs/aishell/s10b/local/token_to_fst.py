#!/usr/bin/env python3

# Copyright 2020 Mobvoi AI Lab, Beijing, China (author: Fangjun Kuang)
# Apache 2.0

# This program takes as input a phone symbol table
# `tokens.txt` and prints a text fst to the console.
#
# You can use `fstcompile` to convert the printed text fst
# to a binary fst.
#
# Two integer values in the symbol table have particular meaning:
#  - 0 for `<eps>`
#  - 1 for the blank symbol `<blk>`

import argparse
import os


def get_args():
    parser = argparse.ArgumentParser(description='''
Convert tokens.txt to tokens.fst.

Usage:
    python3 ./local/token_to_fst.py \
            --tokens-txt-filename data/lang/tokens.txt |
    fstcompile \
      --isymbols=data/lang/tokens.txt \
      --osymbols=data/lang/tokens.txt \
      --keep_isymbols=false \
      --keep_osymbols=false |
    fstarcsort --sort_type=olabel > $data/lang/T.fst || exit 1
''')

    parser.add_argument('--tokens-txt-filename',
                        dest='tokens_txt_filename',
                        help="a phone symbol table",
                        type=str)

    args = parser.parse_args()
    assert os.path.isfile(args.tokens_txt_filename)

    return args


def main():
    args = get_args()

    s = '0 1 <eps> <eps>\n'
    s += '1 1 <blk> <eps>\n'
    s += '2 2 <blk> <eps>\n'
    s += '2 0 <eps> <eps>\n'

    next_state = 3
    with open(args.tokens_txt_filename, 'r') as f:
        for line in f:
            phone_index = line.split()
            assert len(phone_index) == 2
            phone, _ = phone_index

            if phone in ['<eps>', '<blk>']:
                continue

            if '#' in phone:
                s += '0 0 <eps> {}\n'.format(phone)
                continue

            s += '1 {next_state} {phone} {phone}\n'.format(
                next_state=next_state, phone=phone)

            s += '{next_state} {next_state} {phone} <eps>\n'.format(
                next_state=next_state, phone=phone)

            s += '{next_state} 2 <eps> <eps>\n'.format(next_state=next_state)

            next_state += 1

    s += '0'
    print(s)


if __name__ == '__main__':
    main()
