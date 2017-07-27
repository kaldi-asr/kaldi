#!/usr/bin/env python

import argparse
import os
import sys

parser = argparse.ArgumentParser(description="""Creates the list of characters and words in lexicon""")
parser.add_argument('database_path', type=str, help='path to text file')
parser.add_argument('dir', type=str, help='output path')
args = parser.parse_args()

### main ###
char = {}
lex = {}

text_path = os.path.join(args.database_path,'text')
with open(text_path) as f:
  for line in f:
    line = line.strip()
    line_vect = line.split(' ')
    for i in range(1,len(line_vect)):
      characters = list(line_vect[i])
      entry = " ".join(characters)

      entry = entry.replace("#", "<HASH>")

      lex[line_vect[i]] = entry
#      for c in characters:
#        char[c] = " "

#char_file = os.path.join(args.dir, 'nonsilence_phones.txt')
#char_fh = open(char_file, 'w+')

lex_file = os.path.join(args.dir, 'lexicon.txt')
lex_fh = open(lex_file, 'w+')

#char_count = 0
#for key in sorted(char):
#  #char_fh.write(key + " " + str(char_count) + "\n")
#  char_fh.write(key + "\n")
#  char_count = char_count + 1

for key in sorted(lex):
  lex_fh.write(key + " " + lex[key] + "\n")
