#!/bin/bash

# Copyright 2016  Allen Guo
# Apache License 2.0

# This script creates the data directories that will be used during training.
# This is discussed fully in README.md, but the gist of it is that the data for each
# stage will be located at data/{MULTI}/{STAGE}, and every training stage is individually
# configurable for maximum flexibility.

# Note: The $stage if-blocks use -eq in this script, so running with --stage 4 will
# run only the stage 4 prep.

multi=multi_a  # This defines the "variant" we're using; see README.md
stage=1

. utils/parse_options.sh

data_dir=data/$multi
tmp1=$data_dir/tmp1
tmp2=$data_dir/tmp2

if [ $stage -eq 1 ]; then
  utils/subset_data_dir.sh --first data/wsj/train 7138 $tmp1  # tmp1 is si-84
  utils/subset_data_dir.sh --shortest $tmp1 2000 $data_dir/mono
fi

if [ $stage -eq 2 ]; then
  utils/subset_data_dir.sh --first data/wsj/train 7138 $tmp1  # tmp1 is si-84
  utils/subset_data_dir.sh $tmp1 3500 $data_dir/mono_ali
  ln -nfs mono_ali $data_dir/tri1
fi

if [ $stage -eq 3 ]; then
  ln -nfs ../wsj/train $data_dir/tri1_ali
  ln -nfs tri1_ali $data_dir/tri2
fi

if [ $stage -eq 4 ]; then
  utils/subset_data_dir.sh data/swbd/train \
    83000 $tmp1  # tmp1 is ~100 hr of swbd data
  utils/subset_data_dir.sh data/fisher/train \
    99000 $tmp2  # tmp2 is ~100 hr of fisher data
  utils/combine_data.sh $data_dir/tri2_ali \
    $tmp1 $tmp2 data/{ami_ihm,wsj,tedlium,librispeech_100}/train \
    || { echo "Failed to combine data"; exit 1; }
  ln -nfs tri2_ali $data_dir/tri3
fi

if [ $stage -eq 5 ]; then
  utils/combine_data.sh $data_dir/tri3_ali \
    data/{wsj,swbd,fisher,ami_ihm,tedlium,librispeech_100,librispeech_360}/train \
    || { echo "Failed to combine data"; exit 1; }
  ln -nfs tri3_ali $data_dir/tri4
fi

if [ $stage -eq 6 ]; then
  utils/combine_data.sh $data_dir/tri4_ali \
    data/{wsj,swbd,fisher,ami_ihm,tedlium,librispeech_100,librispeech_360,librispeech_500}/train \
    || { echo "Failed to combine data"; exit 1; }
  ln -nfs tri4_ali $data_dir/tri5
fi

if [ $stage -eq 7 ]; then
  ln -nfs tri5 $data_dir/tri5_ali
  ln -nfs tri5 $data_dir/tdnn
  utils/subset_data_dir.sh $data_dir/tdnn \
    100000 $data_dir/tdnn_100k
  utils/subset_data_dir.sh $data_dir/tdnn \
    30000 $data_dir/tdnn_30k
fi

# remove temporary directories
rm -rf $tmp1 $tmp2
