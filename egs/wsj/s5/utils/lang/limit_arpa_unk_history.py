#!/usr/bin/env python3

# Copyright 2018    Armin Oliya
# Apache 2.0.

'''
This script takes an existing ARPA lanugage model and limits the <unk> history
to make it suitable for downstream <unk> modeling.
This is for the case when you don't have access
to the original text corpus that is used for creating the LM.
If you do, you can use pocolm with the option --limit-unk-history=true.
This keeps the graph compact after adding the unk model.
'''

import argparse
import io
import re
import sys
from collections import defaultdict


parser = argparse.ArgumentParser(
    description='''This script takes an existing ARPA lanugage model
    and limits the <unk> history to make it suitable
    for downstream <unk> modeling.
    It supports up to 5-grams.''',
    usage='''utils/lang/limit_arpa_unk_history.py
    <oov-dict-entry> <input-arpa >output-arpa''',
    epilog='''E.g.: gunzip -c src.arpa.gz |
    utils/lang/limit_arpa_unk_history.py "<unk>" | gzip -c >dest.arpa.gz''')

parser.add_argument(
    'oov_dict_entry',
    help='oov identifier, for example "<unk>"', type=str)
args = parser.parse_args()


def get_ngram_stats(old_lm_lines):
    ngram_counts = defaultdict(int)

    for i in range(10):
        g = re.search(r"ngram (\d)=(\d+)", old_lm_lines[i])
        if g:
            ngram_counts[int(g.group(1))] = int(g.group(2))

    if len(ngram_counts) == 0:
        sys.exit("""Couldn't get counts per ngram section.
            The input doesn't seem to be a valid ARPA language model.""")

    max_ngrams = list(ngram_counts.keys())[-1]
    skip_rows = ngram_counts[1]

    if max_ngrams > 5:
        sys.exit("This script supports up to 5-gram language models.")

    return max_ngrams, skip_rows, ngram_counts


def find_and_replace_unks(old_lm_lines, max_ngrams, skip_rows):
    ngram_diffs = defaultdict(int)
    whitespace_pattern = re.compile("[ \t]+")
    unk_pattern = re.compile(
        "[0-9.-]+(?:[\s\\t]\S+){1,3}[\s\\t]" + args.oov_dict_entry +
        "[\s\\t](?!-[0-9]+\.[0-9]+).*")
    backoff_pattern = re.compile(
        "[0-9.-]+(?:[\s\\t]\S+){1,3}[\s\\t]<unk>[\s\\t]-[0-9]+\.[0-9]+")
    passed_2grams, last_ngram = False, False
    unk_row_count, backoff_row_count = 0, 0

    print("Upadting the language model .. ", file=sys.stderr)
    new_lm_lines = old_lm_lines[:skip_rows]

    for i in range(skip_rows, len(old_lm_lines)):
            line = old_lm_lines[i].strip(" \t\r\n")

            if "\{}-grams:".format(3) in line:
                passed_2grams = True
            if "\{}-grams:".format(max_ngrams) in line:
                last_ngram = True

            for i in range(max_ngrams):
                if "\{}-grams:".format(i+1) in line:
                    ngram = i+1

            # remove any n-gram states of the form: foo <unk> -> X
            # that is, any n-grams of order > 2 where <unk>
            # is the second-to-last word
            # here we skip 1-gram and 2-gram sections of arpa

            if passed_2grams:
                g_unk = unk_pattern.search(line)
                if g_unk:
                    ngram_diffs[ngram] = ngram_diffs[ngram] - 1
                    unk_row_count += 1
                    continue

            # remove backoff probability from the lines that end with <unk>
            # for example, the -0.64 in -4.09 every <unk> -0.64
            # here we skip the last n-gram section because it
            # doesn't include backoff probabilities

            if not last_ngram:
                g_backoff = backoff_pattern.search(line)
                if g_backoff:
                    updated_row = whitespace_pattern.split(g_backoff.group(0))[:-1]
                    updated_row = updated_row[0] + \
                        "\t" + " ".join(updated_row[1:]) + "\n"
                    new_lm_lines.append(updated_row)
                    backoff_row_count += 1
                    continue

            new_lm_lines.append(line+"\n")

    print("Removed {} lines including {} as second-to-last term.".format(
        unk_row_count, args.oov_dict_entry), file=sys.stderr)
    print("Removed backoff probabilties from {} lines.".format(
        backoff_row_count), file=sys.stderr)

    return new_lm_lines, ngram_diffs


def read_old_lm():
    print("Reading ARPA LM frome input stream .. ", file=sys.stderr)

    with io.TextIOWrapper(
            sys.stdin.buffer,
            encoding="latin-1") as input_stream:
        old_lm_lines = input_stream.readlines()

    return old_lm_lines


def write_new_lm(new_lm_lines, ngram_counts, ngram_diffs):
    ''' Update n-gram counts that go in the header of the arpa lm '''

    for i in range(10):
        g = re.search(r"ngram (\d)=(\d+)", new_lm_lines[i])
        if g:
            n = int(g.group(1))
            if n in ngram_diffs:
                # ngram_diffs contains negative values
                new_num_ngrams = ngram_counts[n] + ngram_diffs[n]
                new_lm_lines[i] = "ngram {}={}\n".format(
                    n, new_num_ngrams)

    with io.TextIOWrapper(
            sys.stdout.buffer,
            encoding="latin-1") as output_stream:
        output_stream.writelines(new_lm_lines)


def main():
    old_lm_lines = read_old_lm()
    max_ngrams, skip_rows,  ngram_counts = get_ngram_stats(old_lm_lines)
    new_lm_lines, ngram_diffs = find_and_replace_unks(
        old_lm_lines, max_ngrams, skip_rows)
    write_new_lm(new_lm_lines, ngram_counts, ngram_diffs)


if __name__ == "__main__":
    main()
