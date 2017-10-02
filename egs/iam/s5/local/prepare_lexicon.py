#!/usr/bin/env python

import argparse
import os
import sys

parser = argparse.ArgumentParser(description="""Creates the list of characters and words in lexicon""")
parser.add_argument('database_path', type=str, help='path to train text file')
parser.add_argument('test_text', type=str, help='path to test text file to include it in lexicon')
parser.add_argument('dir', type=str, help='output path')
args = parser.parse_args()

### main ###
char = {}
lex = {}

text_path = os.path.join(args.database_path,'text.txt')
with open(text_path) as f:
  for line in f:
    line = line.strip()
    line_vect = line.split(' ')
    for i in range(1,len(line_vect)):
      characters = list(line_vect[i])
      entry = " ".join(characters)
      entry = entry.replace("#", "<HASH>")
      if line_vect[i]:
        lex[line_vect[i]] = entry

if args.test_text > 1:
  text_path = os.path.join(args.test_text,'text.txt')
  with open(text_path) as f:
    for line in f:
      line = line.strip()
      line_vect = line.split(' ')
      for i in range(1,len(line_vect)):
        characters = list(line_vect[i])
        entry = " ".join(characters)
        entry = entry.replace("#", "<HASH>")
        if line_vect[i]:
          lex[line_vect[i]] = entry


lex_file = os.path.join(args.dir, 'lexicon.txt')
lex_fh = open(lex_file, 'w+')
for key in sorted(lex):
  lex_fh.write(key + " " + lex[key] + "\n")
