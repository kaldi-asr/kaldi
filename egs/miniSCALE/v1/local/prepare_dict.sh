#!/usr/bin/env bash

# Copyright      2017  Hossein Hadian
#                2017  Chun Chieh Chang
#                2017  Ashish Arora

# This script prepares the dictionary.

set -e
dir=data/local/dict
segments=data/train/segmented_words
mkdir -p $dir

cat $segments | tr ' ' '\n' | sort -u | \
  LC_ALL=en_US.UTF-8 awk '{len=split($0,chars,""); printf($0); for (i=0;i<=len;i++){printf(chars[i]" ")}; printf("\n")};' | \
  sed 's/.$//' >  $dir/lexicon.txt || exit 1;

cut -d' ' -f2- $dir/lexicon.txt | tr ' ' '\n' | sort -u >$dir/nonsilence_phones.txt || exit 1;

echo '<sil> SIL' >> $dir/lexicon.txt

echo SIL > $dir/silence_phones.txt

echo SIL >$dir/optional_silence.txt

echo -n "" >$dir/extra_questions.txt
