#!/usr/bin/env python3

# Copyright 2014 Vassil Panayotov
# Apache 2.0

# Removes common strings, that are not helpful for language modeling purposes,
# from the Project Gutenberg's texts

# See: kaldi/egs/librispeech/s5/local/lm/python/pre_filter.py

import argparse
import re
import sys
import string

roman_number = [re.compile('^\s*[_LXVI]+(\.)?\s*$'), ]

sq_brackets = [re.compile('(.*)(\[.+\])(.*)', re.IGNORECASE)]

pipes = [re.compile('^\s*\|.*\|\s*$')]

non_word = [re.compile('^\W+$')]

chapter = [re.compile('^\s*((Chapter)|(Volume)|(Canto)).*[LXIV0-9]+.*$', re.IGNORECASE), ]

contents = [re.compile('CONTENTS'),
            re.compile('^.*((\s{2,50})|([\t]+))[0-9]+\s*$'),
            re.compile('^\s*((I+[:.]+)|(I?[LXV]+I*([\.:])?))\s+.*')]

debug = None


strip_chars = " \t\r\n"
whitespace = re.compile("[ \t]+")
punctuation = r"(^|\s+)([" + re.escape(string.punctuation) + r"]+)(?=(\s+|$))"


def parse_opts():
    parser = argparse.ArgumentParser(
        description='Strips unhelpful, from LM viewpoint, strings from PG texts',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--debug', default=False, action='store_true',
                        help='Debug info - e.g. showing the lines that were stripped')

    parser.add_argument('in_text', type=str, help='Input text file')
    parser.add_argument('out_text', type=str, help='Filtered output text file')
    opts = parser.parse_args()
    global debug
    debug = opts.debug
    return opts


def debug_log(lines, idx, context=2):
    if debug:
        start = max(0, idx - context)
        end = min(len(lines), idx + context + 1)
        sys.stderr.write('\n'.join('> %s' % l for l in lines[start:end]) + '\n\n')


def match(regexes, line):
    for r in regexes:
        if r.match(line) is not None:
            return True
    return False


def empty_lines(lines, index, extent):
    """
    If the 'extent' is negative, the function checks the preceding lines
    """
    if extent > 0:
        start = min(index + 1, len(lines) - 1)
        end = min(index + extent + 1, len(lines))
    else:
        start = max(0, index + extent)
        end = max(0, index)
    for l in lines[start:end]:
        if len(l) > 0:
            return False
    return True


if __name__ == '__main__':
    opts = parse_opts()

    with open(opts.in_text, 'r', encoding="utf-8") as in_text:
        in_lines = [l.strip() for l in in_text.readlines()]

    out_text = open(opts.out_text, 'w', encoding="utf-8")

    for i, l in enumerate(in_lines):
        if len(l) == 0:
            continue

        # Roman numeral alone in a line, surrounded by empty lines
        if match(roman_number, l) and empty_lines(in_lines, i, -1) and empty_lines(in_lines, i, 1):
            # print 'matched roman'
            debug_log(in_lines, i)
            continue

        if match(chapter, l) and (empty_lines(in_lines, i, -1) or empty_lines(in_lines, i, 1)):
            # print 'matched chapter'
            debug_log(in_lines, i)
            continue

        if match(non_word, l):
            debug_log(in_lines, i)
            continue

        if match(contents, l):
            # print 'matched contents'
            debug_log(in_lines, i)
            continue

        if match(pipes, l):
            debug_log(in_lines, i)
            continue

        if match(sq_brackets, l):
            debug_log(in_lines, i)
            l = sq_brackets[0].sub(r'\1\3', l)

        # test nsw_expand by an example
        # echo "Additional information about Purdue can be found at www.purduepharma.com. " | \
        # nsw_expand -format opl /dev/stdin

        # remove the space before single quote ('s, 't, 'd, 've, 'll, 'm)
        l = re.sub(r"(\s+)(\')([a-z]{1,2})", r"\2\3", l)

        # remove the space in the time
        l = re.sub(r"(\d{1,2})(\s*)(\:)(\s*)(\d{2})", r"\1\3\5", l)

        # TODOs:
        # (1) H&M, AT&T
        # (2) 12km / h
        # (3) Toyota RAV4

        # remove space in percentages
        l = re.sub(r"(\d{1})(\s*)(\%)", r"\1\3", l)

        # remove the space in currencies: USD, EUR, JPY/CNY, GBP, KPW, RUB
        l = re.sub(r"([\$€¥£₩₽])(\s*)", r"\1", l)
        # it seems euros is not recognized correctly, because the symbol is not ascii
        # it seems pounds is not recognized if used in this way: £ 12.5m => 12.5 million pounds

        # remove punctuation
        l = re.sub(punctuation, r"\1\3", l)  # remove real punctuations, which are surrounded by spaces
        l = re.sub("\s+", " ", l)  # remove multiple spaces
        l = l.strip(strip_chars)

        out_text.write(l + '\n')

    out_text.close()

