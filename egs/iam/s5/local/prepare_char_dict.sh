#!/usr/bin/env bash

database_path=$1
dir=$2

mkdir -p $dir

local/prepare_char_lexicon.py $database_path $dir

cut -d' ' -f2- $dir/lexicon.txt | tr ' ' '\n' | sort -u >$dir/nonsilence_phones.txt || exit 1;

( echo '<sil> SIL'; ) | cat - $lex >> $dir/lexicon.txt || exit 1;

printf "SIL\n" >$dir/silence_phones.txt

echo SIL >$dir/optional_silence.txt

echo -n "" >$dir/extra_questions.txt
