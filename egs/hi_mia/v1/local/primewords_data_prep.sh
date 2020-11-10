#!/usr/bin/env bash

# Copyright 2019 Xingyu Na
# Apache 2.0

. ./path.sh || exit 1;

if [ $# != 2 ]; then
  echo "Usage: $0 <corpus-path> <data-path>"
  echo " $0 /export/a05/xna/data/primewords data/primewords"
  exit 1;
fi

corpus=$1/primewords_md_2018_set1
data=$2

if [ ! -d $corpus/audio_files ] || [ ! -f $corpus/set1_transcript.json ]; then
  echo "Error: $0 requires complete corpus"
  exit 1;
fi

echo "**** Creating primewords data folder ****"

mkdir -p $data/train

# find wav audio file for train

find $corpus -iname "*.wav" > $data/wav.flist
n=`cat $data/wav.flist | wc -l`
[ $n -ne 50384 ] && \
  echo Warning: expected 50384 data files, found $n

echo "Filtering data using found wav list and provided transcript"
local/primewords_parse_transcript.py $data/wav.flist $corpus/set1_transcript.json $data/train

for file in wav.scp utt2spk; do
  sort $data/train/$file -o $data/train/$file
done
utils/utt2spk_to_spk2utt.pl $data/train/utt2spk > $data/train/spk2utt

rm -r $data/wav.flist

utils/data/validate_data_dir.sh --no-feats --no-text $data/train || exit 1;
