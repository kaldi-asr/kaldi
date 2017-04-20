#!/bin/bash

# Copyright 2015  University of Sheffield (Author: Ning Ma)
#           2017  Ruhr-University Bochum (Author: Hendrik Meutzner)
# Apache 2.0.
#
# This script prepares the data/ directory for the CHiME/GRID corpus.
# Note that speaker 21 is excluded from the data as there are no official video files available.

# Begin configuration section.
include_spkid=true  # if set to true, then include speaker ID in utt2spk. Set this to false if all the utterances are of the same speaker
# End configuration section.
if [ -f path.sh ]; then . ./path.sh; fi
. parse_options.sh || exit 1;

# Setup relevant folders
data="$REC_ROOT/data"
mkdir -p "$data/local"
utils="utils"

# Setup wav folders
wav_train="$WAV_ROOT/train"
wav_devel="$WAV_ROOT/devel"
wav_test="$WAV_ROOT/test"

# create train folder in data/
set_list="train"
mkdir -p "$data/train"

# create devel folder in data/ if respective audio files exist
if [ -d "$wav_devel" ]; then
  set_list="$set_list devel"
  mkdir -p "$data/devel"
fi

# create test folder in data/ if respective audio files exist
if [ -d "$wav_test" ]; then
  set_list="$set_list test"
  mkdir -p "$data/test"
fi

echo "Preparing data sets: $set_list"

# Create scp file for train set
scp="$data/train/wav.scp"
rm -f "$scp"
for sid in `seq 34`; do
  sid2=`printf "s%02d" $sid`
  ls -1 $wav_train/*/id$sid/*.wav \
    | sed "s:\(.*\)/\(.*\)/id.*/\(.*\).wav:${sid2}_\3_\2\t\1/\2/id$sid/\3.wav:" \
    | sort >> $scp
done

# Create scp files for devel and test sets
for x in "devel" "test"; do
  if [ -d "$data/$x" ]; then
    scp="$data/$x/wav.scp"
    rm -f "$scp"
    wav_var="wav_$x"
    wav_dir="${!wav_var}"

    for sid in `seq 34`; do
      sid2=`printf "s%02d" $sid`

      ls -1 $wav_dir/*/*/s${sid}_*.wav \
        | sed "s/\(.*\)\/\(.*\)\/s.*_\(.*\).wav/${sid2}_\3_\2\t\1\/\2\/s${sid}_\3.wav/" \
        | sort >> $scp

    done
  fi
done

# Remove speaker 21 as it is not contained in the original video data of GRID
for x in $set_list; do
  # create a backup of audio lists
  cp $data/$x/wav.scp $data/$x/wav.bak
  sed -i '/s21_/d' $data/$x/wav.scp
done

# Prepare other files in data/setname/
for x in $set_list; do
  scp="$data/$x/wav.scp"
  if [ -f "$scp" ]; then
    # Create transcription files
    cut -f1 $scp | local/create_chime1_trans.pl - > "$data/$x/text"

    # Create utt2spk files
    if [ $include_spkid = true ]; then
      # Use speaker ID
      sed "s/\(s..\)\(.*\)[ \t].*/\1\2\t\1/" < "$scp" > "$data/$x/utt2spk"
    else
      # No speaker ID
      sed 's/\(.*\)[ \t].*/\1\t\1/' < "$scp" > "$data/$x/utt2spk"
    fi

    # Create spk2utt files
    cat "$data/$x/utt2spk" | $utils/utt2spk_to_spk2utt.pl > "$data/$x/spk2utt" || exit 1;
  fi
done

echo "--> Data preparation succeeded"
exit 0
