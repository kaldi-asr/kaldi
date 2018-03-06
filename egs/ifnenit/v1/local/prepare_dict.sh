#!/usr/bin/env bash

train_text=$1
test_text=$2
dir=$3

mkdir -p $dir

local/prepare_lexicon_word.py $train_text $test_text $dir

cut -d' ' -f2- $dir/lexicon.txt | tr ' ' '\n' | sort -u >$dir/nonsilence_phones.txt || exit 1;

( echo 'SIL SIL'; ) >> $dir/lexicon.txt || exit 1;

printf "SIL\n" >$dir/silence_phones.txt

echo SIL >$dir/optional_silence.txt

echo -n "" >$dir/extra_questions.txt
