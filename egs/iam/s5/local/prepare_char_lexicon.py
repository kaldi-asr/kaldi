#!/usr/bin/env python

import argparse
import os
import sys

parser = argparse.ArgumentParser(description="""Creates the list of characters and words in lexicon""")
parser.add_argument('database_path', type=str, help='path to text file')
parser.add_argument('dir', type=str, help='output path')
args = parser.parse_args()

### main ###
lex = {}
text_path = os.path.join(args.database_path,'text')
with open(text_path) as f:
  for line in f:
    line = line.strip()
    line_vect = line.split(' ')
    for i in range(1,len(line_vect)):
      characters = list(line_vect[i])
      for c in characters:
        lex[c] = c 
        if c=='#':
          lex[c] = "<HASH>"

lex_file = os.path.join(args.dir, 'lexicon.txt')
lex_fh = open(lex_file, 'w+')

for key in sorted(lex):
  lex_fh.write(key + " " + lex[key] + "\n")
