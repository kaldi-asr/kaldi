#!/usr/bin/env bash

dir_train=$1
dir=$2

mkdir -p $dir

cut -d' ' -f2- $dir_train/text | tr -cs '[a-z][A-Z][0-9][:punct:]' '\n' | sort -u | \
  awk '{len=split($0,chars,""); printf($0); for (i=0;i<=len;i++){if(chars[i]=="#"){chars[i]="<HASH>"} printf(chars[i]" ")}; printf("\n")};' | \
  sed 's/.$//' > $dir/lexicon.txt;

cut -d' ' -f2- $dir/lexicon.txt | tr ' ' '\n' | sort -u >$dir/nonsilence_phones.txt || exit 1;

( echo '<sil> SIL'; ) >> $dir/lexicon.txt || exit 1;
( echo '<unk> NSN'; ) >> $dir/lexicon.txt || exit 1;

( echo SIL; echo NSN ) > $dir/silence_phones.txt

echo SIL >$dir/optional_silence.txt

echo -n "" >$dir/extra_questions.txt
