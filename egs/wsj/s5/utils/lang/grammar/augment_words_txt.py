#!/usr/bin/env python3


import argparse
import os
import sys
import re

def get_args():
    parser = argparse.ArgumentParser(description="""This script augments a words.txt
       file (a word-level symbol table) by adding certain special symbols
       relating to grammar support.  See ../add_nonterminals.sh for context,
       and augment_phones_txt.py.""")

    parser.add_argument('input_words_txt', type=str,
                        help='Filename of input words.txt file, to be augmented')
    parser.add_argument('nonterminal_symbols_list', type=str,
                        help='Filename of a file containing a list of nonterminal '
                        'symbols, one per line.  E.g. #nonterm:contact_list')
    parser.add_argument('output_words_txt', type=str, help='Filename of output '
                        'words.txt file.  May be the same as input-words-txt.')
    args = parser.parse_args()
    return args




def read_words_txt(filename):
    """Reads the words.txt file in 'filename', returns a 2-tuple (lines, highest_symbol)
       where 'lines' is all the lines the words.txt as a list of strings,
       and 'highest_symbol' is the integer value of the highest-numbered symbol
       in the symbol table.  It is an error if the words.txt is empty or mis-formatted."""

    # The use of latin-1 encoding does not preclude reading utf-8.  latin-1
    # encoding means "treat words as sequences of bytes", and it is compatible
    # with utf-8 encoding as well as other encodings such as gbk, as long as the
    # spaces are also spaces in ascii (which we check).  It is basically how we
    # emulate the behavior of python before python3.
    whitespace = re.compile("[ \t]+")
    with open(filename, 'r', encoding='latin-1') as f:
        lines = [line.strip(" \t\r\n") for line in f]
        highest_numbered_symbol = 0
        for line in lines:
            s = whitespace.split(line)
            try:
                i = int(s[1])
                if i > highest_numbered_symbol:
                    highest_numbered_symbol = i
            except:
                raise RuntimeError("Could not interpret line '{0}' in file '{1}'".format(
                line, filename))
            if s[0] in [ '#nonterm_begin', '#nonterm_end' ]:
                raise RuntimeError("It looks like the symbol table {0} already has nonterminals "
                                   "in it.".format(filename))
        return lines, highest_numbered_symbol


def read_nonterminals(filename):
    """Reads the user-defined nonterminal symbols in 'filename', checks that
       it has the expected format and has no duplicates, and returns the nonterminal
       symbols as a list of strings, e.g.
       ['#nonterm:contact_list', '#nonterm:phone_number', ... ]. """
    ans = [line.strip(" \t\r\n") for line in open(filename, 'r', encoding='latin-1')]
    if len(ans) == 0:
        raise RuntimeError("The file {0} contains no nonterminal symbols.".format(filename))
    for nonterm in ans:
        if nonterm[:9] != '#nonterm:':
            raise RuntimeError("In file '{0}', expected nonterminal symbols to start with '#nonterm:', found '{1}'"
                               .format(filename, nonterm))
    if len(set(ans)) != len(ans):
        raise RuntimeError("Duplicate nonterminal symbols are present in file {0}".format(filename))
    return ans

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


def main():
    args = get_args()
    (lines, highest_symbol) = read_words_txt(args.input_words_txt)
    nonterminals = read_nonterminals(args.nonterminal_symbols_list)
    write_words_txt(lines, highest_symbol, nonterminals, args.output_words_txt)


if __name__ == '__main__':
      main()
