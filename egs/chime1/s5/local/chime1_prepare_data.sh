#!/bin/bash

# Copyright 2015  University of Sheffield (Author: Ning Ma)
# Apache 2.0.
#
# This script prepares the data/ directory for the CHiME/GRID corpus

. ./config.sh # Needed for REC_ROOT and WAV_ROOT

# Setup relevant folders
data="$REC_ROOT/data"
locdata="$data/local"
mkdir -p "$locdata"
utils="utils"

# Setup wav folders
wav_train="$WAV_ROOT/train/reverberated"
wav_devel="$WAV_ROOT/devel/isolated"
wav_test="$WAV_ROOT/test/isolated"
if [ ! -d $wav_train ]; then
  echo "Cannot find wav directory $wav_train"
  echo "Please download the CHiME Challenge Data from"
  echo "  train set  http://spandh.dcs.shef.ac.uk/projects/chime/PCC/data/PCCdata16kHz_train_reverberated.tar.gz"
  exit 1;
fi
set_list="train"
mkdir -p "$data/train"
if [ -d "$wav_devel" ]; then
  set_list="$set_list devel"
  mkdir -p "$data/devel"
fi
if [ -d "$wav_test" ]; then
  set_list="$set_list test"
  mkdir -p "$data/test"
fi
echo "Preparing data sets: $set_list"

# Create scp files
scp="$data/train/wav.scp"
rm -f "$scp"
for sid in `seq 34`; do
  sid2=`printf "s%02d" $sid`
  ls -1 $wav_train/id$sid/*.wav \
    | sed "s/\(.*\)\/\(.*\).wav/${sid2}_\2\t\1\/\2.wav/" \
    | sort >> $scp
done
for x in "devel" "test"; do
  if [ -d "$data/$x" ]; then
    scp="$data/$x/wav.scp"
    rm -f "$scp"
    wav_var="wav_$x"
    wav_dir="${!wav_var}"
    for sid in `seq 34`; do
      sid2=`printf "s%02d" $sid`
      ls -1 $wav_dir/*/s${sid}_*.wav \
        | sed "s/\(.*\)\/\(.*\)\/s.*_\(.*\).wav/${sid2}_\3_\2\t\1\/\2\/s${sid}_\3.wav/" \
        | sort >> $scp 
    done
  fi
done

# Prepare other files in data/setname/
for x in $set_list; do
  scp="$data/$x/wav.scp"
  if [ -f "$scp" ]; then
    # Create transcription files
    cut -f1 $scp | local/create_chime1_trans.pl - > "$data/$x/text"

    # Create utt2spk files 
    # No speaker ID
    sed 's/\(.*\)\t.*/\1\t\1/' < "$scp" > "$data/$x/utt2spk"
    # Use speaker ID
    #sed "s/\(s..\)\(.*\)\t.*/\1\2\t\1/" < "$scp" > "$data/$x/utt2spk"

    # Create spk2utt files
    cat "$data/$x/utt2spk" | $utils/utt2spk_to_spk2utt.pl > "$data/$x/spk2utt" || exit 1;
  fi
done

echo "--> Data preparation succeeded"
exit 0
