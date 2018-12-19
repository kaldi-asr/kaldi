#!/usr/bin/env python

# Copyright 2015  Brno University of Technology (author: Karel Vesely)
# Apache 2.0

import sys

from optparse import OptionParser
desc = """
Prepare mapping of words into categories. Each word with minimal frequency 
has its own category, the rest is merged into single class.
"""
usage = "%prog [opts] words.txt ctm category_mapping"
parser = OptionParser(usage=usage, description=desc)
parser.add_option("--min-count", help="Minimum word-count to have a single word category. [default %default]", type='int', default=20)
(o, args) = parser.parse_args()

if len(args) != 3:
  parser.print_help()
  sys.exit(1)
words_file, text_file, category_mapping_file = args

if text_file == '-': text_file = '/dev/stdin'
if category_mapping_file == '-': category_mapping_file = '/dev/stdout'

# Read the words from the 'tra' file,
with open(text_file) as f:
  text_words = [ l.split()[1:] for l in f ]

# Flatten the array of arrays of words,
import itertools
text_words = list(itertools.chain.from_iterable(text_words))

# Count the words (regardless if correct or incorrect),
word_counts = dict()
for w in text_words:
  if w not in word_counts: word_counts[w] = 0
  word_counts[w] += 1

# Read the words.txt,
with open(words_file) as f:
  word_id = [ l.split() for l in f ]

# Append the categories,
n=1
word_id_cat=[]
for word, idx in word_id:
  cat = 0 
  if word in word_counts:
    if word_counts[word] > o.min_count:
      cat = n; n += 1
  word_id_cat.append([word, idx, str(cat)])

# Store the mapping,
with open(category_mapping_file,'w') as f:
  f.writelines([' '.join(record)+'\n' for record in word_id_cat])
