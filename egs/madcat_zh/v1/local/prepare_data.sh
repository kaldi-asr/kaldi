#!/bin/bash

# Copyright      2017  Chun Chieh Chang
#                2017  Ashish Arora
#                2017  Hossein Hadian
# Apache 2.0

# This script downloads the Madcat Chinese handwriting database and prepares the training
# and test data (i.e text, images.scp, utt2spk and spk2utt) by calling process_data.py.
# It also downloads the LOB and Brown text corpora. It downloads the database files
# only if they do not already exist in download directory.

#  Eg. local/prepare_data.sh
#  Eg. text file: 000_a01-000u-00 A MOVE to stop Mr. Gaitskell from
#      utt2spk file: 000_a01-000u-00 000
#      images.scp file: 000_a01-000u-00 data/local/lines/a01/a01-000u/a01-000u-00.png
#      spk2utt file: 000 000_a01-000u-00 000_a01-000u-01 000_a01-000u-02 000_a01-000u-03

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
