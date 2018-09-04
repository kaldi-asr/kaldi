#!/usr/bin/env bash

# Copyright      2017  Hossein Hadian
#                2017  Chun Chieh Chang
#                2017  Ashish Arora

# This script prepares the dictionary.

set -e
dir=data/local/dict
vocab_size=50000
. ./utils/parse_options.sh

mkdir -p $dir

# First get the set of all letters that occur in data/train/text
cat data/train/text | \
  perl -ne '@A = split; shift @A; for(@A) {print join("\n", split(//)), "\n";}' | \
  sort -u | grep -v "|" > $dir/nonsilence_phones.txt

# Now use the pocolm's wordlist which is the most N frequent words in
# in data/train/text and LOB+Brown corpora (dev and test excluded) with their comprising
# letters as their transcription. Only include words that use the above letters.
# (Letter # is replaced with <HASH>)

export letters=$(cat $dir/nonsilence_phones.txt | tr -d "\n")

head -n $vocab_size data/local/local_lm/data/word_count | awk '{print $2}' | \
  perl -e '$letters=$ENV{letters}; $letters=$letters . "|";
while(<>){
    chop;
    $w = $_;
    if($w =~ m/^[$letters]+$/){
      $trans = join(" ", split(//, $w));
      $trans =~ s/#/<HASH>/g;
      $trans =~ s/\|/SIL/g;
      print "$w $trans\n";
    }
}' | sort -u > $dir/lexicon.txt


sed -i "s/#/<HASH>/" $dir/nonsilence_phones.txt

echo '<sil> SIL' >> $dir/lexicon.txt

echo SIL > $dir/silence_phones.txt

echo SIL >$dir/optional_silence.txt

echo -n "" >$dir/extra_questions.txt
