#!/bin/bash

# Copyright 2015  University of Sheffield (Author: Ning Ma)
# Apache 2.0.
#
# Kaldi scripts for preparing dictionary for the GRID corpus (or CHiME 1)

echo "Preparing dictionary"

. ./config.sh # Needed for REC_ROOT and WAV_ROOT

# Prepare relevant folders
dict="$REC_ROOT/data/local/dict"
mkdir -p $dict

utils="utils"

# Copy lexicon
lexicon="input/lexicon.txt" # phone models
cp $lexicon $dict/lexicon.txt

# Generate phone list
sil="SIL"
phone_list="$dict/phone.list" 
awk '{for (n=2;n<=NF;n++)print $n;}' $lexicon | sort -u > $phone_list
echo $sil >> $phone_list

# Create phone lists 
grep -v -w $sil $phone_list > $dict/nonsilence_phones.txt
echo $sil > $dict/silence_phones.txt
echo $sil > $dict/optional_silence.txt

# list of "extra questions"-- empty; we don't  have things like tone or 
# word-positions or stress markings.
touch $dict/extra_questions.txt

echo "-->Dictionary preparation succeeded"
exit 0
