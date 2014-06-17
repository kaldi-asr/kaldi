#!/bin/bash

# Copyright 2010-2012 Microsoft Corporation  Johns Hopkins University (Author: Daniel Povey)
# Copyright 2014 Mirsk Digital ApS  (Author: Andreas Kirkedal)

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
dir=data/local/dict
mkdir -p $dir


# Dictionary preparation:

# Normalise transcripts and create a transcript file
# Removes '.,:;?' and removes '\' before '\Komma' (dictated ',') 
# outputs a normalised transcript without utterance ids and a list of utterance ids 
echo "Normalising"
trainsrc=data/local/trainsrc
mkdir $trainsrc
mv data/train/text1 $trainsrc/text1
python3 local/normalize_transcript_prefixed.py local/norm_dk/numbersUp.tbl $trainsrc/text1 $trainsrc/onlyids $dir/transcripts.tmp

# Additional normalisation, uppercasing, writing numbers etc.
# and recombine with 
local/norm_dk/format_text.sh am $dir/transcripts.tmp > $dir/transcripts.am
cp $dir/transcripts.am $trainsrc/onlytext
paste -d ' ' $trainsrc/onlyids $trainsrc/onlytext > data/train/text 


# lmsents is output by sprak_data_prep.sh and contains
# sentences that are disjoint from the test and dev set 
python3 local/normalize_transcript.py local/norm_dk/numbersUp.tbl data/local/data/lmsents $dir/lmsents.norm
wait

# Create wordlist from the AM transcripts
cat $dir/transcripts.am | tr [:blank:] '\n' | sort -u > $dir/wlist.txt &




# Because training data is read aloud, there are many occurences of the same
# sentence and bias towards the domain. Make a version where  
# the sentences are unique to reduce bias.
local/norm_dk/format_text.sh lm $dir/lmsents.norm > $dir/transcripts.txt
sort -u $dir/transcripts.txt > $dir/transcripts.uniq

# Copy pre-made phone table 
cp local/dictsrc/complexphones.txt $dir/nonsilence_phones.txt

# Copy pre-made lexicon
cp local/dictsrc/lexicon.txt $dir/lexicon.txt


# silence phones, one per line.
echo SIL > $dir/silence_phones.txt
echo SIL > $dir/optional_silence.txt

touch $dir/extra_questions.txt

# Repeat text preparation on test set, but do not add to dictionary
testsrc=data/local/testsrc
mkdir $testsrc
mv data/test/text1 $testsrc/text1
python3 local/normalize_transcript_prefixed.py local/norm_dk/numbersUp.tbl $testsrc/text1 $testsrc/onlyids $testsrc/transcripts.am 
local/norm_dk/format_text.sh am $testsrc/transcripts.am > $testsrc/onlytext
paste -d ' ' $testsrc/onlyids $testsrc/onlytext > data/test/text


# Repeat text preparation on dev set, but do not add to dictionary
devsrc=data/local/devsrc
mkdir $devsrc
mv data/dev/text1 $devsrc/text1
python3 local/normalize_transcript_prefixed.py local/norm_dk/numbersUp.tbl $devsrc/text1 $devsrc/onlyids $devsrc/transcripts.tmp
local/norm_dk/format_text.sh lm $devsrc/transcripts.tmp > data/dev/transcripts.txt
sort -u data/dev/transcripts.txt > data/dev/transcripts.uniq &
local/norm_dk/format_text.sh am $devsrc/transcripts.tmp > $devsrc/onlytext
paste -d ' ' $devsrc/onlyids $devsrc/onlytext > data/dev/text


## TODO: add cleanup commands

echo "Dictionary preparation succeeded"

