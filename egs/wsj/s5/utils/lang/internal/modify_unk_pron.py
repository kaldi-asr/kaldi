#!/usr/bin/env python

# Copyright 2016  Johns Hopkins University (Author: Daniel Povey)
# Apache 2.0.

from __future__ import print_function
import sys
import os
import argparse
from collections import defaultdict

# note, this was originally based

parser = argparse.ArgumentParser(description="""
This script replaces the existing pronunciation of the
unknown word in the provided lexicon, with a pronunciation
consisting of three disambiguation symbols: #1 followed by #2
followed by #3.
The #2 will later be replaced by a phone-level LM by
apply_unk_lm.sh (called later on by prepare_lang.sh).
Caution: this script is sensitive to the basename of the
lexicon: it should be called either lexiconp.txt, in which
case the format is 'word pron-prob p1 p2 p3 ...'
or lexiconp_silprob.txt, in which case the format is
'word pron-prob sil-prob1 sil-prob2 sil-prob3 p1 p2 p3....'.
It is an error if there is not exactly one pronunciation of
the unknown word in the lexicon.""",
epilog="""E.g.: modify_unk_pron.py data/local/lang/lexiconp.txt '<unk>'.
This script is called from prepare_lang.sh.""")

parser.add_argument('lexicon_file', type = str,
                    help = 'Filename of the lexicon file to operate on (this is '
                    'both an input and output of this script).')
parser.add_argument('unk_word', type = str,
                    help = "The printed form of the unknown/OOV word, normally '<unk>'.")

args = parser.parse_args()

if len(args.unk_word.split()) != 1:
    sys.exit("{0}: invalid unknown-word '{1}'".format(
        sys.argv[0], args.unk_word))

basename = os.path.basename(args.lexicon_file)
if basename != 'lexiconp.txt' and basename != 'lexiconp_silprob.txt':
    sys.exit("{0}: expected the basename of the lexicon file to be either "
             "'lexiconp.txt' or 'lexiconp_silprob.txt', got: {1}".format(
                 sys.argv[0], args.lexicon_file))
# the lexiconp.txt format is: word pron-prob p1 p2 p3...
# lexiconp_silprob.txt has 3 extra real-valued fields after the pron-prob.
num_fields_before_pron = 2 if basename == 'lexiconp.txt' else 5

print(' '.join(sys.argv), file = sys.stderr)

try:
    lexicon_in = open(args.lexicon_file, 'r')
except:
    sys.exit("{0}: failed to open lexicon file {1}".format(
        sys.argv[0], args.lexicon_file))

split_lines = []
unk_index = -1
while True:
    line = lexicon_in.readline()
    if line == '':
        break
    this_split_line = line.split()
    if this_split_line[0] == args.unk_word:
        if unk_index != -1:
            sys.exit("{0}: expected there to be exactly one pronunciation of the "
                     "unknown word {1} in {2}, but there are more than one.".format(
                         sys.argv[0], args.lexicon_file, args.unk_word))
        unk_index = len(split_lines)
    if len(this_split_line) <= num_fields_before_pron:
        sys.exit("{0}: input file {1} had a bad line (too few fields): {2}".format(
            sys.argv[0], args.lexicon_file, line[:-1]))
    split_lines.append(this_split_line)

if len(split_lines) == 0:
    sys.exit("{0}: read no data from lexicon file {1}.".format(
        sys.argv[0], args.lexicon_file))


if unk_index == -1:
    sys.exit("{0}: expected there to be exactly one pronunciation of the "
             "unknown word {1} in {2}, but there are none.".format(
                 sys.argv[0], args.unk_word, args.lexicon_file))

lexicon_in.close()

# now modify the pron.
split_lines[unk_index] = split_lines[unk_index][0:num_fields_before_pron] + [ '#1', '#2', '#3' ]


try:
    # write to the same file.
    lexicon_out = open(args.lexicon_file, 'w')
except:
    sys.exit("{0}: failed to open lexicon file {1} for writing (permissions probleM?)".format(
        sys.argv[0], args.lexicon_file))

for split_line in split_lines:
    print(' '.join(split_line), file = lexicon_out)

try:
    lexicon_out.close()
except:
    sys.exit("{0}: failed to close lexicon file {1} after writing (disk full?)".format(
        sys.argv[0], args.lexicon_file))
