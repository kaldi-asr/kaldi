#!/usr/bin/env bash


set -e
dir=data/local/dict

. ./utils/parse_options.sh

mkdir -p $dir

# First get the set of all letters that occur in data/train/text
echo "heysnips" > $dir/nonsilence_phones.txt
echo "freetext" >> $dir/nonsilence_phones.txt

echo "HeySnips heysnips" > $dir/lexicon.txt
echo "FREETEXT freetext" >> $dir/lexicon.txt
echo "<sil> SIL" >> $dir/lexicon.txt

echo SIL > $dir/silence_phones.txt

echo SIL >$dir/optional_silence.txt

echo -n "" >$dir/extra_questions.txt
