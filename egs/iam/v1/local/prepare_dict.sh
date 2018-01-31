#!/usr/bin/env bash

# Copyright      2017  Hossein Hadian
#                2017  Chun Chieh Chang
#                2017  Ashish Arora

# This script prepares the dictionary.

set -e
dir=data/local/dict
mkdir -p $dir

# First get the set of all letters that occur in data/train/text
cat data/train/text | \
  perl -ne '@A = split; shift @A; for(@A) {print join("\n", split(//)), "\n";}' | \
  sort -u > $dir/nonsilence_phones.txt

# Now list all the unique words (that use only the above letters)
# in data/train/text and LOB+Brown corpora with their comprising
# letters as their transcription. (Letter # is replaced with <HASH>)

export letters=$(cat $dir/nonsilence_phones.txt | tr -d "\n")

cut -d' ' -f2- data/train/text | \
  cat data/local/lobcorpus/0167/download/LOB_COCOA/lob.txt \
      data/local/browncorpus/brown.txt - | \
  perl -e '$letters=$ENV{letters};
while(<>){ @A = split;
  foreach(@A) {
    if(! $seen{$_} && $_ =~ m/^[$letters]+$/){
      $seen{$_} = 1;
      $trans = join(" ", split(//));
      $trans =~ s/#/<HASH>/g;
      print "$_ $trans\n";
    }
  }
}' | sort > $dir/lexicon.txt


sed -i '' "s/#/<HASH>/" $dir/nonsilence_phones.txt

echo '<sil> SIL' >> $dir/lexicon.txt
echo '<unk> SIL' >> $dir/lexicon.txt

echo SIL > $dir/silence_phones.txt

echo SIL >$dir/optional_silence.txt

echo -n "" >$dir/extra_questions.txt
