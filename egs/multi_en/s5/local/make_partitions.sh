#!/bin/bash

# Copyright 2016  Allen Guo
#           2017  Xiaohui Zhang
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

mkdir -p $data_dir
# swbd 10k short (nodup)
if [ $stage -eq 1 ]; then
  utils/subset_data_dir.sh --shortest data/swbd/train 100000 data/swbd/train_100kshort
  utils/data/remove_dup_utts.sh 10 data/swbd/train_100kshort data/swbd/train_100kshort_nodup
  utils/subset_data_dir.sh  data/swbd/train_100kshort_nodup  10000 data/swbd/train_10k_nodup
  ln -nfs ../swbd/train_10k_nodup $data_dir/mono
fi

# swbd 30k (nodup)
if [ $stage -eq 2 ]; then
  utils/subset_data_dir.sh --speakers data/swbd/train 30000 data/swbd/train_30k
  utils/data/remove_dup_utts.sh 200 data/swbd/train_30k $data_dir/mono_ali
  ln -nfs mono_ali $data_dir/tri1a
  ln -nfs mono_ali $data_dir/tri1a_ali
  ln -nfs mono_ali $data_dir/tri1b
fi

# swbd 100k (nodup)
if [ $stage -eq 3 ]; then
  utils/subset_data_dir.sh --speakers data/swbd/train 100000 data/swbd/train_100k
  utils/data/remove_dup_utts.sh 200 data/swbd/train_100k $data_dir/train_100k_nodup
  ln -nfs train_100k_nodup $data_dir/tri1b_ali
  ln -nfs train_100k_nodup $data_dir/tri2
fi

# whole swbd
if [ $stage -eq 4 ]; then
  ln -nfs ../swbd/train $data_dir/tri2_ali
  ln -nfs tri2_ali $data_dir/tri3a
fi

# whole fisher + swbd (nodup)
if [ $stage -eq 5 ]; then
  utils/combine_data.sh data/fisher_swbd \
    data/{swbd,fisher}/train \
    || { echo "Failed to combine data"; exit 1; }
  utils/data/remove_dup_utts.sh 300 data/fisher_swbd $data_dir/tri3a_ali
  ln -nfs tri3a_ali $data_dir/tri3b
fi


# whole fisher + swbd + tedlium (nodup)
if [ $stage -eq 6 ]; then
  utils/combine_data.sh $data_dir/fisher_swbd_tedlium \
    data/{fisher,swbd}/train $data_dir/tedlium_cleaned/train \
    || { echo "Failed to combine data"; exit 1; }
  utils/data/remove_dup_utts.sh 300 $data_dir/fisher_swbd_tedlium $data_dir/tri3b_ali
  ln -nfs tri3b_ali $data_dir/tri4
fi

# whole fisher + swbd + tedlium + wsj + hub4_en (nodup)
if [ $stage -eq 7 ]; then
  utils/combine_data.sh $data_dir/fisher_swbd_tedlium_wsj_hub4 \
    $data_dir/fisher_swbd_tedlium data/{wsj,hub4_en}/train \
    || { echo "Failed to combine data"; exit 1; }
  utils/data/remove_dup_utts.sh 300 $data_dir/fisher_swbd_tedlium_wsj_hub4 $data_dir/tri4_ali
  ln -nfs tri4_ali $data_dir/tri5a
fi

# whole fisher + swbd + tedlilum + wsj + hub4_en + librispeech460 (nodup)
if [ $stage -eq 8 ]; then
  utils/combine_data.sh $data_dir/fisher_swbd_tedlium_wsj_hub4_libri460 \
    $data_dir/fisher_swbd_tedlium_wsj_hub4 data/{librispeech_100,librispeech_360}/train \
    || { echo "Failed to combine data"; exit 1; }
  utils/data/remove_dup_utts.sh 300 $data_dir/fisher_swbd_tedlium_wsj_hub4_libri460 $data_dir/tri5a_ali
  ln -nfs tri5a_ali $data_dir/tri5b
fi

# whole fisher + swbd + tedlilum + wsj + hub4_en + librispeech960 (nodup)
if [ $stage -eq 9 ]; then
  utils/combine_data.sh $data_dir/fisher_swbd_tedlium_wsj_hub4_libri960 \
    $data_dir/fisher_swbd_tedlium_wsj_hub4_libri460 data/librispeech_500/train \
    || { echo "Failed to combine data"; exit 1; }
  utils/data/remove_dup_utts.sh 300 $data_dir/fisher_swbd_tedlium_wsj_hub4_libri960 $data_dir/tri5b_ali
  ln -nfs tri5b_ali $data_dir/tri6a
  ln -nfs tri5b_ali $data_dir/tri6a_ali
fi

# sampled data for ivector extractor training,.etc
if [ $stage -eq 10 ]; then
  ln -nfs tri6a $data_dir/tdnn
  utils/subset_data_dir.sh $data_dir/tdnn \
    100000 $data_dir/tdnn_100k
  utils/subset_data_dir.sh $data_dir/tdnn \
    30000 $data_dir/tdnn_30k
fi

