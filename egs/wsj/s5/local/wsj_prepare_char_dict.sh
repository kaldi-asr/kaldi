#!/usr/bin/env bash

# Copyright 2017  Hossein Hadian

phone_dir=data/local/dict_nosp
dir=data/local/dict_char
mkdir -p $dir

[ -f path.sh ] && . ./path.sh

# Simply transcribe each word with its comprising characters:

# We keep only one pronunciation for each word. Other alternative pronunciations are discarded.
cat $phone_dir/lexicon1_raw_nosil.txt | \
  perl -e 'while(<>){@A = split; if(! $seen{$A[0]}) {$seen{$A[0]} = 1; print $_;}}' \
       > $phone_dir/lexicon2_raw_nosil.txt || exit 1;


cat $phone_dir/lexicon2_raw_nosil.txt | python -c 'import sys
for l in sys.stdin:
  w = l.strip().split(" ")[0]
  r = w
  for c in w:
    if c not in "!~@#$%^&*()+=/\",;:?_{}-":
      r += " " + c
  print r
' > $dir/lexicon2_raw_nosil.txt || exit 1;

(echo SIL; echo SPN; echo NSN) > $dir/silence_phones.txt
echo SIL > $dir/optional_silence.txt

(echo '!SIL SIL'; echo '<SPOKEN_NOISE> SPN'; \
 echo '<UNK> SPN'; echo '<NOISE> NSN'; ) | \
 cat - $dir/lexicon2_raw_nosil.txt | sort | uniq > $dir/lexicon.txt || exit 1;

#  Get the set of non-silence phones
cut -d' ' -f2- $dir/lexicon2_raw_nosil.txt | tr ' ' '\n' | \
  sort -u > $dir/nonsilence_phones.txt

echo "Character-based dictionary preparation succeeded."
