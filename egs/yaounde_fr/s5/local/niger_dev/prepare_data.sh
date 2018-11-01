#!/bin/bash

# Copyright 2018 John Morgan
# Apache 2.0.

# niger dev prep

if [ $# != 1 ]; then
  echo "usage: $0 <CORPUS_DIRECTORY>
example:
$0 African_AccentedFrench";
  exit 1
fi

# set variables
datadir=$1
speech_datadir=$datadir/speech/dev/niger_west_african_fr
tmpdir=data/local/tmp/niger
# done setting variables

mkdir -p $tmpdir
#get a list of the niger .wav files
find $speech_datadir -type f -name "*.wav" > $tmpdir/wav_list.txt
#  make niger dev lists
local/niger_dev/make_lists.pl $datadir
utils/fix_data_dir.sh $tmpdir/lists
mkdir -p data/dev
for x in spk2utt text utt2spk wav.scp; do
  cp $tmpdir/lists/$x data/dev/
done
