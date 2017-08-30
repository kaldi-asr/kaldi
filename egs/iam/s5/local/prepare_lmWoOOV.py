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
#lex = {}
#text_path = os.path.join(args.database_path,'text')
#with open(text_path) as f:
#  for line in f:
#    line = line.strip()
#    line_vect = line.split(' ')
#    for i in range(1,len(line_vect)):
#      characters = list(line_vect[i])
#      entry = " ".join(characters)
#      entry = entry.replace("#", "<HASH>")
#      lex[line_vect[i]] = entry

#text_path = os.path.join(args.test_text,'text')
#with open(text_path) as f:
#  for line in f:
#    line = line.strip()
#    line_vect = line.split(' ')
#    for i in range(1,len(line_vect)):
#      lex[line_vect[i]] = line_vect[i]
#
#lex_file = os.path.join(args.database_path,'text')
#lex_fh = open(lex_file, 'a+')
#for key in sorted(lex):
#  lex_fh.write(lex[key] + " " + key + "\n")

### main ###
lex = {}
text_path = os.path.join(args.database_path,'text')
with open(text_path) as f:
  for line in f:
    line = line.strip()
    line_vect = line.split(' ')
    lex[line_vect[0]] = line

text_path = os.path.join(args.test_text,'text')
with open(text_path) as f:
  for line in f:
    line = line.strip()
    line_vect = line.split(' ')
    for i in range(1,len(line_vect)):
      line_vect[i] = line_vect[i] + " " + line_vect[i]
      lex[line_vect[i]] = line_vect[i]

lex_file = os.path.join(args.dir,'train_test')
lex_fh = open(lex_file, 'w+')
for key in sorted(lex):
  lex_fh.write(lex[key] + "\n")
