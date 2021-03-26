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

local/prepare_lexicon.py --data-dir $data_dir $dir

perl -i -ne 'print if /\S/' $dir/lexicon.txt
cut -d' ' -f2- $dir/lexicon.txt | sed 's/SIL//g' | tr ' ' '\n' | sort -u | sed '/^$/d' >$dir/nonsilence_phones.txt || exit 1;

echo '<sil> SIL' >> $dir/lexicon.txt

echo SIL > $dir/silence_phones.txt

echo SIL >$dir/optional_silence.txt

echo -n "" >$dir/extra_questions.txt
