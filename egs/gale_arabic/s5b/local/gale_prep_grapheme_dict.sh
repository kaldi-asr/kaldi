#!/bin/bash

# Copyright 2017 QCRI (author: Ahmed Ali)
# Apache 2.0


# run this from ../
dir=$(utils/make_absolute.sh data/local/dict)
mkdir -p $dir


# (1) Get all avaialble  dictionaries, since this is a grapheme model, so we mainly need the most frequent word lists
wget http://alt.qcri.org//resources/speech/dictionary/ar-ar_grapheme_lexicon_2016-02-09.bz2  || exit 1;
wget http://alt.qcri.org//resources/speech/dictionary/ar-ar_lexicon_2014-03-17.txt.bz2  || exit 1;
bzcat ar-ar_grapheme_lexicon_2016-02-09.bz2  | sed '1,3d' | awk '{print $1}'  >  tmp$$
bzcat ar-ar_lexicon_2014-03-17.txt.bz2 | sed '1,3d' | awk '{print $1}' >>  tmp$$
# (2) Now we add all the words appeared in the training data
cat data/local/train/text | cut -d ' ' -f 2- | tr -s " " "\n" | sort -u >> tmp$$
grep -v [0-9] tmp$$ |  sed -e 's:[FNKaui\~o\`]::g' -e 's:{:}:g' | sort -u > tmp1.$$ # remove vowels and  rare alef wasla
cat tmp1.$$ | sed 's:\(\):\1 :g' | sed -e 's:  : :g' -e 's:  : :g' -e 's:\s*: :g' -e  's:\*:V:g' > tmp2.$$
paste -d ' ' tmp1.$$ tmp2.$$ > $dir/lexicon.txt 

#(2) Dictionary preparation:

# silence phones, one per line.
echo SIL > $dir/silence_phones.txt
echo SIL > $dir/optional_silence.txt

# nonsilence phones; on each line is a list of phones that correspond
# really to the same base phone.
cat tmp2.$$ | tr -s ' ' '\n' | grep -v ^$  | sort -u >  $dir/nonsilence_phones.txt || exit 1;

sed -i '1i<UNK> SIL' $dir/lexicon.txt # insert word <UNK> with phone sil at the begining of the dictionary

rm -fr ar-ar_lexicon_2014-03-17.txt.bz2 ar-ar_grapheme_lexicon_2016-02-09.bz2 tmp$$ tmp1.$$ tmp2.$$ 
echo Dictionary preparation succeeded

# The script is still missing dates and numbers 

exit 0 

