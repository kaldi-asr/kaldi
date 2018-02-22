#!/bin/bash

# Copyright      2017  Chun Chieh Chang
#                2017  Hossein Hadian

set -e
dir=data/local/dict

mkdir -p $dir

cut -d' ' -f2- data/train/text | tr -cs '[a-z][A-Z][0-9][:punct:]' '\n' | sort -u | \
  awk '{len=split($0,chars,""); printf($0);
       for (i=0;i<=len;i++) {
         if(chars[i]=="#") {chars[i]="<HASH>"}
         printf(chars[i]" ")
       };
       printf("\n")};' | \
  sed 's/.$//' > $dir/lexicon.txt;

cut -d' ' -f2- $dir/lexicon.txt | tr ' ' '\n' | sort -u >$dir/nonsilence_phones.txt

echo '<sil> SIL' >> $dir/lexicon.txt
echo '<unk> SIL' >> $dir/lexicon.txt

echo SIL > $dir/silence_phones.txt

echo SIL > $dir/optional_silence.txt

echo -n "" > $dir/extra_questions.txt
