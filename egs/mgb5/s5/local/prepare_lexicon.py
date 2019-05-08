#!/usr/bin/env python3

# Copyright      2018  Ashish Arora
# Apache 2.0

# This script prepares lexicon.

import argparse
import os

parser = argparse.ArgumentParser(description="""Creates the list of characters and words in lexicon""")
args = parser.parse_args()

### main ###
lex = {}
text_path = os.path.join('data','local', 'lexicon_data', 'processed_lexicon')
with open(text_path, 'r', encoding='utf-8') as f:
    for line in f:
        line = line.strip()
        characters = list(line)
        characters = " ".join(['V' if char == '*' else char for char in characters])
        lex[line] = characters

with open(os.path.join('data','local','dict', 'lexicon.txt'), 'w', encoding='utf-8') as fp:
    for key in sorted(lex):
        fp.write(key + "  " + lex[key] + "\n")
