#!/bin/bash

# Copyright 2017 Xingyu Na
# Apache 2.0

. ./path.sh || exit 1;

if [ $# != 2 ]; then
  echo "Usage: $0 <audio-path> <text-path>"
  echo " $0 /export/a05/xna/data/data_aishell/wav /export/a05/xna/data/data_aishell/transcript"
  exit 1;
fi

aishell_audio_dir=$1
aishell_text_dir=$2

train_dir=data/local/train
dev_dir=data/local/dev
test_dir=data/local/test

mkdir -p $train_dir
mkdir -p $dev_dir
mkdir -p $test_dir

# data directory check
if [ ! -d $aishell_audio_dir ] || [ ! -d $aishell_text_dir ]; then
  echo "Error: $0 requires two directory arguments"
  exit 1;
fi

# find wav audio file for train, dev and test resp.
find $aishell_audio_dir -iname "*.wav" | grep -i "wav/train" > $train_dir/wav.flist || exit 1;
find $aishell_audio_dir -iname "*.wav" | grep -i "wav/dev" > $dev_dir/wav.flist || exit 1;
find $aishell_audio_dir -iname "*.wav" | grep -i "wav/test" > $test_dir/wav.flist || exit 1;

n=`cat $train_dir/wav.flist $dev_dir/wav.flist $test_dir/wav.flist | wc -l`
[ $n -ne 141925 ] && \
  echo Warning: expected 141925 data data files, found $n

# Transcriptions preparation
for dir in $train_dir $test_dir; do
  echo Preparing $dir transcriptions
  sed -e 's/\.wav//' $dir/wav.flist | awk -F '/' '{print $NF}' |\
    sort > $dir/utt.list
  sed -e 's/\.wav//' $dir/wav.flist | awk -F '/' '{i=NF-1;printf("%s %s\n",$NF,$i)}' |\
    sort > $dir/utt2spk_all
  paste -d' ' $dir/utt.list $dir/wav.flist > $dir/wav.scp_all
  utils/filter_scp.pl -f 1 $dir/utt.list $aishell_text_dir/*.txt > $dir/transcripts.txt
  awk '{print $1}' $dir/transcripts.txt > $dir/utt.list
  utils/filter_scp.pl -f 1 $dir/utt.list $dir/utt2spk_all | sort -u > $dir/utt2spk
  utils/filter_scp.pl -f 1 $dir/utt.list $dir/wav.scp_all | sort -u > $dir/wav.scp
  sort -u $dir/transcripts.txt > $dir/text
  utils/utt2spk_to_spk2utt.pl $dir/utt2spk > $dir/spk2utt
done

mkdir -p data/train data/test
for f in spk2utt utt2spk wav.scp text; do
  cp $train_dir/$f data/train/$f || exit 1;
  cp $test_dir/$f data/test/$f || exit 1;
done

echo "$0: AISHELL data preparation succeeded"
exit 0;
