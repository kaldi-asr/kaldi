#!/usr/bin/env bash

# Copyright      2017  Chun Chieh Chang
#                2017  Ashish Arora
#                2017  Hossein Hadian

# This script prepares the dictionary based on the training words.

set -e
dir=data/local/dict
mkdir -p $dir

# List all the unique words in data/train/text with their comprising
# letters as their transcription. Letter # is replaced with <HASH>.
cat data/train/text |  perl -ne '@A = split; shift @A;
  foreach(@A) {
    if(! $seen{$_}){
      $seen{$_} = 1;
      $trans = join(" ", split(//));
      $trans =~ s/#/<HASH>/g;
      print "$_ $trans\n";
    }
  }' | sort > $dir/lexicon.txt


cut -d' ' -f2- $dir/lexicon.txt | tr ' ' '\n' | sort -u >$dir/nonsilence_phones.txt

( echo '<sil> SIL'; ) >> $dir/lexicon.txt || exit 1;
( echo '<unk> SIL'; ) >> $dir/lexicon.txt || exit 1;

( echo SIL ) > $dir/silence_phones.txt

echo SIL >$dir/optional_silence.txt

echo -n "" >$dir/extra_questions.txt
