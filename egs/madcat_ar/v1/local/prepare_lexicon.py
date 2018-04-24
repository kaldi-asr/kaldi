#!/usr/bin/env python3

# Copyright  2018  Ashish Arora

import argparse
import os
import sys

parser = argparse.ArgumentParser(description="""Creates the list of characters and words in lexicon""")
parser.add_argument('dir', type=str, help='output path')
args = parser.parse_args()

### main ###
lex = {}
text_path = os.path.join('data', 'train', 'text')
text_fh = open(text_path, 'r', encoding='utf-8')

for line in text_fh:
    line_vect = line.strip().split(' ')
    for i in range(1,len(line_vect)):
        characters = list(line_vect[i])
        characters = " ".join(characters)
        lex[line_vect[i]] = characters
        if line_vect[i] =='#':
            lex[line_vect[i]] = "<HASH>"

lex_file = os.path.join(args.dir, 'lexicon.txt')
lex_fh = open(lex_file, 'w+', encoding='utf-8')

for key in sorted(lex):
  lex_fh.write(key + " " + lex[key] + "\n")
