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
echo "Normalising "
python3 local/normalize_transcript_prefixed.py local/norm_dk/numbersUp.tbl data/train/text1 data/train/onlyids $dir/transcripts.tmp

# Additional normalisation, uppercasing, writing numbers etc.
# and recombine with 
local/norm_dk/format_text.sh am $dir/transcripts.tmp > $dir/transcripts.am
cp $dir/transcripts.am data/train/onlytext
paste data/train/onlyids data/train/onlytext > data/train/text 


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


## TODO: add cleanup commands

echo "Dictionary preparation succeeded"

