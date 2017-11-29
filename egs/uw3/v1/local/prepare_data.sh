#!/bin/bash

# This scripts loads the UW3 dataset

dir=data

. ./cmd.sh
if [ -f path.sh ]; then . ./path.sh; fi
. parse_options.sh || exit 1;

# Download dir
dl_dir=$dir/download
download_url=http://www.tmbdev.net/ocrdata/uw3-lines-book.tgz
data_dir=book

mkdir -p $dl_dir

if [ -d $dl_dir/$data_dir ]; then
  echo Not downloading dataset as it is already downloaded.
else
  if [ ! -f $dl_dir/uw3-lines-book.tgz ]; then
    echo Downloading dataset...
    wget -P $dl_dir $download_url || exit 1;
  fi
  tar -xzf $dl_dir/uw3-lines-book.tgz -C $dl_dir || exit 1;
  echo Done downloading datset
fi

mkdir -p $dir/train
mkdir -p $dir/test
local/process_data.py $dl_dir/book $dir || exit 1

utils/utt2spk_to_spk2utt.pl $dir/train/utt2spk > $dir/train/spk2utt
utils/utt2spk_to_spk2utt.pl $dir/test/utt2spk > $dir/test/spk2utt
