#!/usr/bin/env bash

# Copyright 2019 QCRI (author: Ahmed Ali)
# Apache 2.0
# This script prepares the grapaheme dictionary

set -e
dir=data/local/dict
lexicon_url1="https://arabicspeech.org/static/data_resources/ar-ar_grapheme_lexicon_20160209.bz2";
lexicon_url2="https://arabicspeech.org/static/data_resources/ar-ar_phoneme_lexicon_20140317.bz2";
stage=0
. ./cmd.sh
. ./path.sh
. ./utils/parse_options.sh || exit 1;
mkdir -p $dir data/local/lexicon_data

if [ $stage -le 0 ]; then
  echo "$0: Downloading text for lexicon... $(date)."
  if [ ! -f data/local/lexicon_data/ar-ar_grapheme_lexicon_20160209.bz2 ]; then
    wget -P data/local/lexicon_data $lexicon_url1
  else
    echo "data/local/lexicon_data/ar-ar_grapheme_lexicon_20160209.bz2 already exist on disk"
  fi 
  
  if [ ! -f data/local/lexicon_data/ar-ar_phoneme_lexicon_20140317.bz2 ]; then
    wget -P data/local/lexicon_data $lexicon_url2
  else
    echo "data/local/lexicon_data/ar-ar_phoneme_lexicon_20140317.bz2 already exist on disk"
  fi 
  
  rm -fr data/local/lexicon_data/grapheme_lexicon
  for dict in ar-ar_grapheme_lexicon_20160209.bz2 ar-ar_phoneme_lexicon_20140317.bz2; do
    bzcat data/local/lexicon_data/$dict | sed '1,3d' | \
    awk '{print $1}'  >>  data/local/lexicon_data/grapheme_lexicon
  done
  cat data/train/text | cut -d ' ' -f 2- | tr -s " " "\n" | grep -v UNK |  sort -u >> data/local/lexicon_data/grapheme_lexicon
fi


if [ $stage -le 0 ]; then
  echo "$0: processing lexicon text and creating lexicon... $(date)."
  # remove vowels and  rare alef wasla
  grep -v [0-9] data/local/lexicon_data/grapheme_lexicon |  sed -e 's:[FNKaui\~o\`]::g' -e 's:{:}:g' | sort -u > data/local/lexicon_data/processed_lexicon
  local/prepare_lexicon.py
fi

cut -d' ' -f2- $dir/lexicon.txt | sed 's/SIL//g' | tr ' ' '\n' | sort -u | sed '/^$/d' >$dir/nonsilence_phones.txt || exit 1;

sed -i '1i<UNK> UNK' $dir/lexicon.txt

echo UNK >> $dir/nonsilence_phones.txt

echo '<sil> SIL' >> $dir/lexicon.txt

echo SIL > $dir/silence_phones.txt

echo SIL >$dir/optional_silence.txt

echo -n "" >$dir/extra_questions.txt

echo "$0: Dictionary preparation succeeded"
