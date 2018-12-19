#!/usr/bin/env python3

# Copyright      2017  Babak Rekabdar
#                2017  Hossein Hadian
#                2017  Chun Chieh Chang
#                2017  Ashish Arora
# Apache 2.0

# This script prepares lexicon for BPE. It gets the set of all words that occur in data/train/text.
# Since this lexicon is based on BPE, it replaces '|' with silence.

import argparse
import os

parser = argparse.ArgumentParser(description="""Creates the list of characters and words in lexicon""")
parser.add_argument('dir', type=str, help='output path')
args = parser.parse_args()

### main ###
lex = {}
text_path = os.path.join('data', 'train', 'text')
with open(text_path, 'r', encoding='utf-8') as f:
    for line in f:
        line_vect = line.strip().split(' ')
        for i in range(1, len(line_vect)):
            characters = list(line_vect[i])
            characters = " ".join([ 'SIL' if char == '|' else char for char in characters])
            characters = list(characters)
            characters = "".join([ '<HASH>' if char == '#' else char for char in characters])
            lex[line_vect[i]] = characters

with open(os.path.join(args.dir, 'lexicon.txt'), 'w', encoding='utf-8') as fp:
    for key in sorted(lex):
        fp.write(key + " " + lex[key] + "\n")
