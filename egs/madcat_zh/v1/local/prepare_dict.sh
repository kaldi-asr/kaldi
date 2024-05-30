#!/usr/bin/env bash

# Copyright      2017  Hossein Hadian
#                2017  Chun Chieh Chang
#                2017  Ashish Arora

# This script prepares the dictionary.

set -e
dir=data/local/dict
mkdir -p $dir

#local/prepare_lexicon.py data/train $dir
cat data/train/text | cut -d' ' -f2- | tr ' ' '\n' | sort -u | sed '/^$/d' | \
  python3 -c \
  'import sys, io; \
  sys.stdin = io.TextIOWrapper(sys.stdin.buffer, encoding="utf8"); \
  sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf8"); \
  [sys.stdout.write(line.strip() + " " + " ".join(list(line.strip())) + "\n") for line in sys.stdin];' > $dir/lexicon.txt

cut -d' ' -f2- $dir/lexicon.txt | tr ' ' '\n' | sort -u >$dir/nonsilence_phones.txt || exit 1;

echo '<sil> SIL' >> $dir/lexicon.txt

echo SIL > $dir/silence_phones.txt

echo SIL >$dir/optional_silence.txt

echo -n "" >$dir/extra_questions.txt
