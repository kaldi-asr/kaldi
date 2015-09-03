#!/bin/bash

# Copyright 2014 QCRI (author: Ahmed Ali)
# Apache 2.0


# run this from ../
dir=data/local/dict
mkdir -p $dir


# (1) Get QCRI dictionary
wget http://alt.qcri.org//resources/speech/dictionary/ar-ar_lexicon_2014-03-17.txt.bz2  || exit 1;
bzcat ar-ar_lexicon_2014-03-17.txt.bz2 | sed '1,3d'  >  $dir/lexicon.txt 
rm -fr ar-ar_lexicon_2014-03-17.txt.bz2

#(2) Dictionary preparation:

# silence phones, one per line.
echo SIL > $dir/silence_phones.txt
echo SIL > $dir/optional_silence.txt

# nonsilence phones; on each line is a list of phones that correspond
# really to the same base phone.
cat $dir/lexicon.txt | cut -d ' ' -f2- | tr -s ' ' '\n' |\
sort -u >  $dir/nonsilence_phones.txt || exit 1;


 sed -i '1i<UNK> SIL' $dir/lexicon.txt
 
echo Dictionary preparation succeeded

