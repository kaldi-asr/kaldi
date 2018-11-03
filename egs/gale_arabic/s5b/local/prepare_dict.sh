#!/bin/bash

# This script prepares the dictionary.

set -e
dir=data/local/dict
. ./utils/parse_options.sh || exit 1;

mkdir -p $dir

local/prepare_lexicon.py $dir

cut -d' ' -f2- $dir/lexicon.txt | sed 's/SIL//g' | tr ' ' '\n' | sort -u | sed '/^$/d' >$dir/nonsilence_phones.txt || exit 1;

echo '<sil> SIL' >> $dir/lexicon.txt

echo SIL > $dir/silence_phones.txt

echo SIL >$dir/optional_silence.txt

echo -n "" >$dir/extra_questions.txt
exit 0
