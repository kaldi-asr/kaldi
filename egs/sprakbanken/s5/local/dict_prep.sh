#!/usr/bin/env bash

# Copyright 2010-2012 Microsoft Corporation  Johns Hopkins University (Author: Daniel Povey)
# Copyright 2014 Mirsk Digital ApS  (Author: Andreas Kirkedal)
# Copyright 2014-2016 Andreas Kirkedal5D

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

KALDI_ROOT=$(pwd)/../../..

exproot=$(pwd)
lmdir=data/local/transcript_lm
dictsrc=data/local/dictsrc
dictdir=data/local/dict
espeakdir='espeak-1.48.04-source'
mkdir -p $dictsrc $dictdir


# Dictionary preparation:

# Create wordlist from the AM transcripts
cat $lmdir/transcripts.uniq | tr [:blank:] '\n' | sort -u > $dictsrc/wlist.txt &

# Install eSpeak if it is not installed already
if hash espeak 2>/dev/null;
then
    echo 'eSpeak installed'
else
    cd $KALDI_ROOT/tools || exit 1;
    wget http://sourceforge.net/projects/espeak/files/espeak/espeak-1.48/${espeakdir}.zip
    wait
    unzip -q $espeakdir.zip
    cd $espeakdir/src
    # Remove dependency to portaudio - we only need the text-to-phoneme system
    perl -pi.back -e 's/^(AUDIO = portaudio)$/\#\1/' -e 's/^\#(AUDIO = portaudio2)$/\#\1/' Makefile
    make || exit 1;
    echo 'Installed eSpeak'
    cd $exproot || exit 1;
fi

# Wait for the wordlist to be fully created
wait

# Run wordlist through espeak to get phonetics
# improvised parallelisation - simple call because 'split' often has different versions
split -l 10000 $dictsrc/wlist.txt $dictsrc/Wtemp_
for w in $dictsrc/Wtemp_*; do
    (cat $w | espeak -q -vda -x > $w.pho) &
done

wait

cat $dictsrc/Wtemp_*.pho > $dictsrc/plist.txt
rm -f $dictsrc/Wtemp_*


# Filter transcription
# Remove diacritics, language annotation ((da), (en), (fr) etc.), insert space between symbols, remove
# initial and trailing spaces and collapse 2 or more spaces to one space

cat $dictsrc/plist.txt | perl -pe 's/\([[a-z]{2}\)//g' | perl -pe 's// /g' | perl -pe 's/ a I / aI /g' | perl -pe 's/ d Z / dZ /g' | perl -pe 's/ \? / /g' | perl -pe 's/ ([\#]) /\+ /g' | perl -pe 's/([\@n3]) \- /\1\- /g' | perl -pe "s/[\_\:\!\'\,\|2]//g" | perl -pe 's/ \- / /g' | tr -s ' ' | perl -pe 's/^ +| +$//g' > $dictsrc/plist2.txt

#Some question marks are not caught above
perl -pe 's/ \? / /g' $dictsrc/plist2.txt > $dictsrc/plist3.txt

# Create lexicon.txt and put it in data/local/dict
paste $dictsrc/wlist.txt $dictsrc/plist3.txt > $dictsrc/lexicon1.txt

# Remove entries without transcription
grep -P  "^.+\t.+$" $dictsrc/lexicon1.txt > $dictsrc/lexicon2.txt

# Copy pre-made phone table with
cp local/dictsrc/complexphones.txt $dictdir/nonsilence_phones.txt


# Add "!SIL SIL" to lexicon.txt
echo -e '!SIL\tSIL' > $dictsrc/lex_first
echo -e '<UNK>\tSPN' >> $dictsrc/lex_first
cat $dictsrc/lexicon2.txt >> $dictsrc/lex_first
mv $dictsrc/lex_first $dictdir/lexicon.txt

# silence phones, one per line.

if [ ! -f $dictdir/silence_phones.txt ]; then
    echo SIL > $dictdir/silence_phones.txt
fi

if [ ! -f $dictdir/optional_silence.txt ]; then
    echo SIL > $dictdir/optional_silence.txt
fi

if [ ! -f $dictdir/extra_questions.txt ]; then
    touch $dictdir/extra_questions.txt
fi


echo "Dictionary preparation succeeded"
