#!/bin/bash

# Copyright 2010-2012 Microsoft Corporation  Johns Hopkins University (Author: Daniel Povey)

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#  http://www.apache.org/licenses/LICENSE-2.0
#
# THIS CODE IS PROVIDED *AS IS* BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, EITHER EXPRESS OR IMPLIED, INCLUDING WITHOUT LIMITATION ANY IMPLIED
# WARRANTIES OR CONDITIONS OF TITLE, FITNESS FOR A PARTICULAR PURPOSE,
# MERCHANTABLITY OR NON-INFRINGEMENT.
# See the Apache 2 License for the specific language governing permissions and
# limitations under the License.

# Call this script from one level above, e.g. from the s3/ directory.  It puts
# its output in data/local/.

# The parts of the output of this that will be needed are
# [in data/local/dict/ ]
# lexicon.txt
# extra_questions.txt
# nonsilence_phones.txt
# optional_silence.txt
# silence_phones.txt

# run this from ../
#dir=data/local/dict
dir=$2
mkdir -p $dir


# Dictionary preparation:


# Normalise transcripts and create a transcript file
# Removes '.,:;?' and removes '\' before '\Komma' (dictated ',') 
# outputs a normalised transcript without utterance ids
python3 local/normalize_transcript.py data/train/text1 data/train/text $dir/transcripts

# Create wordlist
cat $1 | perl -pe 's/\.$//g' | tr ' ' '\n' | sort -u > $dir/wlist.txt

# Run through espeak to get phonetics
#cat $dir/wlist.txt | espeak -q -vda -x > $dir/plist.txt

split -l 50000 $dir/wlist.txt $dir/Wtemp_
for w in $dir/Wtemp_*; do
  cat $w | espeak -q -vda -x > $w.pho;
#  rm -f $w;
done
cat $dir/Wtemp_*.pho > $dir/plist.txt
rm -f $dir/Wtemp_*

# Filter transcription
# Remove diacritics, language annotation ((da), (en), (fr) etc.), insert space between symbols, remove 
# initial and trailing spaces and collapse 2 or more spaces to one space
cat $dir/plist.txt | tr '^%,=:_|#\$12;-?' ' ' | tr "'" " " | perl -pe 's/\(..\)|\-|\~//g' | perl -pe 's// /g' | perl -pe 's/^ +| +$//g' | tr -s ' ' > $dir/plist2.txt

# Map phones with few occurences (B, Y, L, J, z, U, T, "Z" and x, *) to 
# phones with many occurences (b, y, l, y, s, w, t, dZ and dZ, R respectively)
cat $dir/plist2.txt | tr 'BYLJzUT*Q' 'bylyswtRg' | perl -pe 's/d Z/dZ/g' | perl -pe 's/ ?x ?| Z ?|Z / dZ /g' > $dir/plist3.txt

# Create lexicon.txt and put it in data/local/dict
paste $dir/wlist.txt $dir/plist3.txt > $dir/lexicon1.txt

# Remove entries without transcription
grep -P  "^.+\t.+$" $dir/lexicon1.txt > $dir/lexicon2.txt

# Create nonsilence_phones.txt and put in in data/local/dict
cat $dir/plist3.txt | tr ' ' '\n' | sort -u > $dir/nonsilence_phones1.txt

grep -v "^$" $dir/nonsilence_phones1.txt > $dir/nonsilence_phones.txt

# Add "!SIL SIL" to lexicon.txt
echo -e '!SIL\tSIL' > $dir/lex_first
echo -e '<UNK>\tSPN' >> $dir/lex_first
cat $dir/lexicon2.txt >> $dir/lex_first
mv $dir/lex_first $dir/lexicon.txt

# silence phones, one per line.
(echo SIL; echo SPN) > $dir/silence_phones.txt
echo SIL > $dir/optional_silence.txt

touch $dir/extra_questions.txt


echo "Dictionary preparation succeeded"

