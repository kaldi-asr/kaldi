#!/bin/bash

# Copyright      2018  Ashish Arora
# Apache 2.0

# This script downloads data splits for MADCAT Chinese dataset.
# It also check if madcat chinese data is present or not.

download_dir1=/export/corpora/LDC/LDC2014T13/data
train_split_url=http://www.openslr.org/resources/50/madcat.train.raw.lineid
test_split_url=http://www.openslr.org/resources/50/madcat.test.raw.lineid
dev_split_url=http://www.openslr.org/resources/50/madcat.dev.raw.lineid
data_split_dir=data/download/datasplits

. ./cmd.sh
. ./path.sh
. ./utils/parse_options.sh || exit 1;

if [ -d $data_split_dir ]; then
  echo "$0: Not downloading the data splits as it is already there."
else
  if [ ! -f $data_split_dir/madcat.train.raw.lineid ]; then
    mkdir -p $data_split_dir
    echo "$0: Downloading the data splits..."
    wget -P $data_split_dir $train_split_url || exit 1;
    wget -P $data_split_dir $test_split_url || exit 1;
    wget -P $data_split_dir $dev_split_url || exit 1;
  fi
  echo "$0: Done downloading the data splits"
fi

if [ -d $download_dir1 ]; then
  echo "$0: madcat chinese data directory is present."
else
  if [ ! -f $download_dir1/madcat/*.madcat.xml ]; then
    echo "$0: please download madcat data..."
  fi
fi
