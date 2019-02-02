#!/usr/bin/env python3

# Copyright  2018  Ashish Arora

import argparse
import os

parser = argparse.ArgumentParser(description="""Creates the list of characters and words in lexicon""")
parser.add_argument('dir', type=str, help='output path')
parser.add_argument("--build-bpe-based-dict", action="store_true",
                   help="If true, build a bpe based lexicon")
args = parser.parse_args()

### main ###
lex = {}
text_path = os.path.join('data', 'train', 'text')
text_fh = open(text_path, 'r', encoding='utf-8')

with open(text_path, 'r', encoding='utf-8') as f:
    for line in f:
        line_vect = line.strip().split(' ')
        for i in range(1, len(line_vect)):
            characters = list(line_vect[i])
            if args.build_bpe_based_dict:
                # Put SIL instead of "|". Because every "|" in the beginning of the words is for initial-space of that word
                characters = " ".join(['SIL' if char == '|' else char for char in characters])
            else:
                characters = " ".join(characters)
            lex[line_vect[i]] = characters
            if line_vect[i] == '#':
                lex[line_vect[i]] = "<HASH>"

with open(os.path.join(args.dir, 'lexicon.txt'), 'w', encoding='utf-8') as fp:
    for key in sorted(lex):
        fp.write(key + " " + lex[key] + "\n")
