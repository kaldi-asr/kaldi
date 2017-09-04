#!/usr/bin/env python

import argparse
import os
import sys
import numpy as np
from scipy import misc
import xml.dom.minidom as minidom

parser = argparse.ArgumentParser(description="Creates the list of characters and words in lexicon")
parser.add_argument('database_path', type=str, help='path to train text file')
parser.add_argument('test_text', type=str, help='path to test text file to include it in lexicon')
parser.add_argument('dir', type=str, help='output path')
args = parser.parse_args()

### main ###
lex = {}
train_words = {}
text_path = os.path.join(args.database_path,'text')
with open(text_path) as f:
  for line in f:
    line = line.strip()
    line_vect = line.split(' ')
    lex[line_vect[0]] = line

with open(text_path) as f:
  for line in f:
    line = line.strip()
    line_vect = line.split(' ')
    for i in range(1,len(line_vect)):
      train_words[line_vect[i]] = line_vect[i]

text_path = os.path.join(args.test_text,'text')
with open(text_path) as f:
  for line in f:
    line = line.strip()
    line_vect = line.split(' ')
    for i in range(1,len(line_vect)):
      if line_vect[i] in train_words:
        continue
      else: 
        line_vect[i] = "zz_id " + line_vect[i]
        lex[line_vect[i]] = line_vect[i]

lex_file = os.path.join(args.dir,'train_test')
lex_fh = open(lex_file, 'w+')
for key in sorted(lex):
  lex_fh.write(lex[key] + "\n")
