#!/bin/bash

# Copyright 2018 John Morgan
# Apache 2.0.

# yaounde  answers prep

if [ $# != 1 ]; then
  echo "usage: $0 <CORPUS_DIRECTORY>
example:
$0 African_AccentedFrench";
  exit 1
fi

# set variables
datadir=$1
speech_datadir=$datadir/speech/train/yaounde/answers
tmpdir=data/local/tmp/yaounde_answers
# done setting variables

mkdir -p $tmpdir
#get a list of the yaounde answers .wav files
find $speech_datadir -type f -name "*.wav" | grep answers > $tmpdir/wav_list.txt
#  make yaounde answers lists
local/yaounde_answers/make_lists.pl $datadir
utils/fix_data_dir.sh $tmpdir/lists

mkdir -p data/unsup
for x in spk2utt text utt2spk wav.scp; do
  cp $tmpdir/lists/$x data/unsup/
done
