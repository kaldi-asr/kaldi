#!/bin/bash

# Copyright 2018 John Morgan
# Apache 2.0.

# ca16 read train prep

if [ $# != 1 ]; then
  echo "usage: $0 <CORPUS_DIRECTORY>
example:
$0 African_Accented_French";
  exit 1
fi

# set variables
datadir=$1
speech_datadir=$datadir/speech/train/ca16
tmpdir=data/local/tmp/ca16read_train
# done setting variables

mkdir -p $tmpdir
#get a list of the ca16 read train .wav files
find $speech_datadir -type f -name "*.wav" | grep read > $tmpdir/wav_list.txt
#  make ca16 read train lists
local/ca16_read_train/make_lists.pl $datadir
utils/fix_data_dir.sh $tmpdir/lists
