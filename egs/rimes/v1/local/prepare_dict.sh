#!/usr/bin/env bash

# Copyright      2017  Hossein Hadian
#                2017  Chun Chieh Chang
#                2017  Ashish Arora

# This script prepares the dictionary.

set -e
dir=data/local/dict
build_bpe_based_dict=true
. ./utils/parse_options.sh || exit 1;

mkdir -p $dir
local/prepare_lexicon.py $dir

if $build_bpe_based_dict; then
  local/prepare_lexicon.py $dir --build-bpe-based-dict
  cut -d' ' -f2- $dir/lexicon.txt | sed 's/SIL//g' | tr ' ' '\n' | sort -u | sed '/^$/d' >$dir/nonsilence_phones.txt || exit 1;
else
  local/prepare_lexicon.py $dir
  cut -d' ' -f2- $dir/lexicon.txt | tr ' ' '\n' | sort -u >$dir/nonsilence_phones.txt || exit 1;
  echo '<unk> SIL' >> $dir/lexicon.txt
fi

echo '<sil> SIL' >> $dir/lexicon.txt

echo SIL > $dir/silence_phones.txt

echo SIL >$dir/optional_silence.txt

echo -n "" >$dir/extra_questions.txt
