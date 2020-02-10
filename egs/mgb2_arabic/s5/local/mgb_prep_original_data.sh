#!/usr/bin/env bash

# Copyright (C) 2016, Qatar Computing Research Institute, HBKU


if [ $# -ne 2 ]; then
  echo "Usage: $0 <wav-dir> <text-dir>"
  exit 1;
fi

wavDir=$1
textDir=$2

trainDirOrig=data/train_orig

rm -r $trainDirOrig

for file in train; do
  if [ ! -f $file ]; then
    echo "$0: no such file $file - copy $file from GitHub repository ArabicASRChallenge2016/download/"
    exit 1;
  fi
done 

set -e -o pipefail

mkdir -p $trainDirOrig
cut -d '/' -f 2 train | while read basename; do
  [ ! -e $textDir/$basename.txt ] && echo "Missing $textDir/$basename.txt" && exit 1
  text=$(cat $textDir/$basename.txt | tr '\n' ' ' | perl -pe 's/\s+/ /g')
  [ -z "$text" ] && exit 1
  echo "$basename $text" >> $trainDirOrig/text
  echo "$basename $wavDir/$basename.wav" >> $trainDirOrig/wav.scp
done

awk '{print $1" "$1" 1"}' $trainDirOrig/wav.scp > $trainDirOrig/reco2file_and_channel
awk '{print $1" "$1}' $trainDirOrig/wav.scp > $trainDirOrig/utt2spk
cp $trainDirOrig/utt2spk $trainDirOrig/spk2utt

utils/fix_data_dir.sh $trainDirOrig
utils/validate_data_dir.sh --no-feats $trainDirOrig
