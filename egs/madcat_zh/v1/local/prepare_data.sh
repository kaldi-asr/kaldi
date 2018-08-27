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

stage=0
download_dir=/export/corpora/LDC/LDC2014T13
data_split_dir=data/download/datasplits

. ./cmd.sh
. ./path.sh
. ./utils/parse_options.sh || exit 1;

if [[ ! -d $download_dir ]]; then
  echo "$0: Warning: Couldn't find $download_dir."
  echo ""
fi

mkdir -p data/{train,test,dev}/lines
if [ $stage -le 1 ]; then
  local/process_data.py $download_dir $data_split_dir/madcat.train.raw.lineid data/train || exit 1
  local/process_data.py $download_dir $data_split_dir/madcat.test.raw.lineid data/test || exit 1
  local/process_data.py $download_dir $data_split_dir/madcat.dev.raw.lineid data/dev || exit 1

  for dataset in train test dev; do
    echo "$0: Fixing data directory for dataset: $dataset"
    echo "Date: $(date)."
    image/fix_data_dir.sh data/$dataset
  done
fi
