#!/bin/bash

# Copyright 2017 QCRI (author: Ahmed Ali)
# Apache 2.0

mkdir -p data/local/dict
dir=$(utils/make_absolute.sh data/local/dict)

wget http://alt.qcri.org//resources/speech/dictionary/ar-ar_grapheme_lexicon_2016-02-09.bz2  || exit 1;
wget http://alt.qcri.org//resources/speech/dictionary/ar-ar_lexicon_2014-03-17.txt.bz2  || exit 1;
bzcat ar-ar_grapheme_lexicon_2016-02-09.bz2  | sed '1,3d' | awk '{print $1}'  >  tmp$$
bzcat ar-ar_lexicon_2014-03-17.txt.bz2 | sed '1,3d' | awk '{print $1}' >>  tmp$$

cat data/train/text | cut -d ' ' -f 2- | tr -s " " "\n" | sort -u >> tmp$$
grep -v [0-9] tmp$$ |  sed -e 's:[FNKaui\~o\`]::g' -e 's:{:}:g' | sort -u > tmp1.$$ # remove vowels and  rare alef wasla
cat tmp1.$$ | sed 's:\(\):\1 :g' | sed -e 's:  : :g' -e 's:  : :g' -e 's:\s*: :g' -e  's:\*:V:g' > tmp2.$$
paste -d ' ' tmp1.$$ tmp2.$$ > $dir/lexicon.txt 

sed -i '1i<UNK> SIL' $dir/lexicon.txt

echo SIL > $dir/silence_phones.txt

echo SIL >$dir/optional_silence.txt

echo -n "" >$dir/extra_questions.txt

cat tmp2.$$ | tr -s ' ' '\n' | grep -v ^$  | sort -u >  $dir/nonsilence_phones.txt || exit 1;

rm -fr ar-ar_lexicon_2014-03-17.txt.bz2 ar-ar_grapheme_lexicon_2016-02-09.bz2 tmp$$ tmp1.$$ tmp2.$$ 

echo Dictionary preparation succeeded

exit 0 
