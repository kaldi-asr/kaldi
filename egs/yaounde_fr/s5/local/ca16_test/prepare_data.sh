#!/bin/bash

# Copyright 2018 John Morgan
# Apache 2.0.

# ca16 test prep

if [ $# != 1 ]; then
  echo "usage: $0 <CORPUS_DIRECTORY>
example:
$0 African_Accented_French";
  exit 1
fi

# set variables
datadir=$1
speech_datadir=$datadir/speech/test/ca16
tmpdir=data/local/tmp/ca16_test
# done setting variables

mkdir -p $tmpdir
#get a list of the ca16 test .wav files
find $speech_datadir -type f -name "*.wav" > $tmpdir/wav_list.txt
#  make ca16 test lists
local/ca16_test/make_lists.pl $datadir
utils/utt2spk_to_spk2utt.pl $tmpdir/lists/utt2spk > $tmpdir/lists/spk2utt
mkdir -p data/test
for x in spk2utt text utt2spk wav.scp; do
  cp $tmpdir/lists/$x data/test/
done
utils/fix_data_dir.sh data/test
