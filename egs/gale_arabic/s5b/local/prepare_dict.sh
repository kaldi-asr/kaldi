#!/bin/bash
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
#cat data/train/text | cut -d ' ' -f 2- | tr -s " " "\n" | sort -u >> tmp$$
#grep -v [0-9] tmp$$ |  sed -e 's:[FNKaui\~o\`]::g' -e 's:{:}:g' | sort -u > tmp1.$$ # remove vowels and  rare alef wasla
#cat tmp1.$$ | sed 's:\(\):\1 :g' | sed -e 's:  : :g' -e 's:  : :g' -e 's:\s*: :g' -e  's:\*:V:g' > tmp2.$$
#paste -d ' ' tmp1.$$ tmp2.$$ > $dir/lexicon.txt
#
#cat tmp2.$$ | tr -s ' ' '\n' | grep -v ^$  | sort -u >  $dir/nonsilence_phones.txt || exit 1;
