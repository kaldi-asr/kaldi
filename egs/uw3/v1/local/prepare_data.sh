#!/bin/bash

# Copyright 2017 Chun Chieh Chang

# This script downloads the UW3 dataset (if not already downloaded)
# and prepares the "train" and "test" data subsets.

set -e
download_dir=data/download

. ./cmd.sh
if [ -f path.sh ]; then . ./path.sh; fi
. parse_options.sh || exit 1;

# Download dir
download_url=http://www.tmbdev.net/ocrdata/uw3-lines-book.tgz
data_dir=data/local/extracted_corpus

mkdir -p $download_dir
mkdir -p $data_dir

if [ -d $data_dir/book ]; then
  echo "$0: Not downloading dataset as it is already downloaded."
else
  if [ ! -f $download_dir/uw3-lines-book.tgz ]; then
    echo "$0: Downloading dataset..."
    wget -P $download_dir $download_url || exit 1;
  fi
  echo "$0: Extracting..."
  tar -xzf $download_dir/uw3-lines-book.tgz -C $data_dir/ || exit 1;
  echo "$0: Done downloading/extracting the datset."
fi

mkdir -p data/train
mkdir -p data/test
echo "$0: Preparing the test and train subsets..."
local/process_data.py $data_dir/book data || exit 1

utils/utt2spk_to_spk2utt.pl data/train/utt2spk > data/train/spk2utt
utils/utt2spk_to_spk2utt.pl data/test/utt2spk > data/test/spk2utt
