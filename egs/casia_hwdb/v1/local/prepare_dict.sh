#!/usr/bin/env bash

# Copyright      2017  Hossein Hadian
#                2017  Chun Chieh Chang
#                2017  Ashish Arora

# This script prepares the dictionary.

set -e
dir=data/local/dict
data_dir=data

. ./utils/parse_options.sh || exit 1;

base_dir=$(echo "$DIRECTORY" | cut -d "/" -f2)

mkdir -p $dir

cut -d' ' -f1 download/cj5-cc.txt | ./utils/lang/bpe/learn_bpe.py -s 300 > $dir/bpe.out
cut -d' ' -f1 download/cj5-cc.txt | ./utils/lang/bpe/apply_bpe.py -c $dir/bpe.out | sed 's/@@//g' > $dir/bpe_text
cut -d' ' -f2- download/cj5-cc.txt | sed 's/ //g' > $dir/ids
paste -d' ' $dir/bpe_text $dir/ids > $dir/cj5-cc.txt
local/prepare_lexicon.py --data-dir $data_dir $dir

cut -d' ' -f2- $dir/lexicon.txt | sed 's/SIL//g' | tr ' ' '\n' | sort -u | sed '/^$/d' >$dir/nonsilence_phones.txt || exit 1;

echo '<sil> SIL' >> $dir/lexicon.txt

echo SIL > $dir/silence_phones.txt

echo SIL >$dir/optional_silence.txt

echo -n "" >$dir/extra_questions.txt
