#!/usr/bin/env python3

# Copyright  2018  Ashish Arora

import argparse
import os

parser = argparse.ArgumentParser(description="""Creates the list of characters and words in lexicon""")
parser.add_argument('dir', type=str, help='output path')
parser.add_argument('--data-dir', type=str, default='data', help='Path to text file')
args = parser.parse_args()

### main ###
lex = {}
text_path = os.path.join(args.data_dir, 'train', 'text')
text_fh = open(text_path, 'r', encoding='utf-8')

with open(text_path, 'r', encoding='utf-8') as f:
    for line in f:
        line_vect = line.strip().split(' ')
        for i in range(1, len(line_vect)):
            characters = list(line_vect[i])
	    # Put SIL instead of "|". Because every "|" in the beginning of the words is for initial-space of that word
            characters = " ".join([ 'SIL' if char == '|' else char for char in characters])
            characters = characters.replace('#','<HASH>')
            lex[line_vect[i]] = characters

with open(os.path.join(args.dir, 'lexicon.txt'), 'w', encoding='utf-8') as fp:
    for key in sorted(lex):
        fp.write(key + " " + lex[key] + "\n")
