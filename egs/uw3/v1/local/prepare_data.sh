#!/bin/bash

# Copyright 2017 (Author: Chun Chieh Chang)

# This scripts loads the UW3 dataset

dir=data
download_dir=data/download

. ./cmd.sh
if [ -f path.sh ]; then . ./path.sh; fi
. parse_options.sh || exit 1;

# Download dir
download_url=http://www.tmbdev.net/ocrdata/uw3-lines-book.tgz
data_dir=book

mkdir -p $download_dir

if [ -d $download_dir/$data_dir ]; then
  echo Not downloading dataset as it is already downloaded.
else
  if [ ! -f $download_dir/uw3-lines-book.tgz ]; then
    echo Downloading dataset...
    wget -P $download_dir $download_url || exit 1;
  fi
  tar -xzf $download_dir/uw3-lines-book.tgz -C $download_dir || exit 1;
  echo Done downloading datset
fi

mkdir -p $dir/train
mkdir -p $dir/test
local/process_data.py $download_dir/$data_dir $dir || exit 1

utils/utt2spk_to_spk2utt.pl $dir/train/utt2spk > $dir/train/spk2utt
utils/utt2spk_to_spk2utt.pl $dir/test/utt2spk > $dir/test/spk2utt
