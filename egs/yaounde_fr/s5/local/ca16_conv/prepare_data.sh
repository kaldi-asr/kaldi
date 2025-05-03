#!/bin/bash

# Copyright 2018 John Morgan
# Apache 2.0.

# set variables
datadir=$1
speech_datadir=$datadir/speech/train/ca16
tmpdir=data/local/tmp/ca16conv_train
# end  setting variables

mkdir -p $tmpdir
find $speech_datadir -type f -name "*.wav" | grep  conv > $tmpdir/wav_list.txt
local/ca16_conv/make_lists.pl $datadir
utils/utt2spk_to_spk2utt.pl $tmpdir/lists/
utils/fix_data_dir.sh $tmpdir/lists
