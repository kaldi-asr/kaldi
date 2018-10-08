#!/bin/bash

stage=0


. ./cmd.sh
. ./path.sh
. utils/parse_options.sh

set -euo pipefail

if [ $stage -le 0 ]; then
  local/mobvoi_data_download.sh
  echo "Extracted all datasets into data/download/"
fi

if [ $stage -le 1 ]; then
  echo "Splitting datasets..."
  local/split_datasets.sh
  echo "text and utt2spk have been generated in data/{train|dev|eval}."
fi
    
if [ $stage -le 2 ]; then
  echo "Preparing wav.scp..."
  python3 local/prepare_wav.py data
  echo "wav.scp has been generated in data/{train|dev|eval}."
fi

if [ $stage -le 3 ]; then
  for folder in train dev eval; do
    dir=data/$folder
    utils/fix_data_dir.sh $dir
    steps/make_mfcc.sh --cmd "$train_cmd" --nj 16 $dir
    steps/compute_cmvn_stats.sh $dir
    utils/fix_data_dir.sh $dir
    utils/validate_data_dir.sh $dir
  done
fi
