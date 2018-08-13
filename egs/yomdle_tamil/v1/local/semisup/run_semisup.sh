#!/bin/bash

# Copyright 2017  Vimal Manohar
#           2018  Ashish Arora
# Apache 2.0

# This script demonstrates semi-supervised training using 25k line images of 
# supervised data and 22k line images of unsupervised data.
# We assume the supervised data is in data/train and unsupervised data
# is in data/train_unsup. 
# For LM training, we use 5 million lines of tamil text.

set -e
set -o pipefail
stage=0
nj=30
exp_root=exp/semisup_100k
. ./cmd.sh
. ./path.sh
. ./utils/parse_options.sh

if [ $stage -le 0 ]; then
  echo "stage 0: Processing train unsupervised data...$(date)"
  local/prepare_unsup_data.sh --language tamil
fi

if [ $stage -le 1 ]; then
  for set in train train_unsup; do
    echo "$0: Extracting features and calling compute_cmvn_stats for dataset:  $set. "
    echo "Date: $(date)."
    local/extract_features.sh --nj $nj --cmd $cmd --feat-dim 40 data/${set}
    steps/compute_cmvn_stats.sh data/${set} || exit 1;
    #image/ocr/extract_features.sh --nj $nj --cmd $cmd --feat-dim 40 data/$dataset
    #image/ocr/make_features.py data/$set/images.scp --feat-dim 40 \
    #  --allowed_len_file_path data/$set/allowed_lengths.txt --no-augment | \
    #  copy-feats --compress=true --compression-method=7 \
    #    ark:- ark,scp:data/$set/data/images.ark,data/$set/feats.scp
    #steps/compute_cmvn_stats.sh data/$set || exit 1;
  done
  utils/fix_data_dir.sh data/train

  local/make_features.py data/test/images.scp --feat-dim 40 \
      --allowed_len_file_path data/test/allowed_lengths.txt  --no-augment | \
      copy-feats --compress=true --compression-method=7 \
               ark:- ark,scp:data/test/data/images.ark,data/test/feats.scp
fi

for f in data/train/utt2spk data/train_unsup/utt2spk \
  data/train/text; do
  if [ ! -f $f ]; then
    echo "$0: Could not find $f"
    exit 1
  fi
done
# Prepare semi-supervised train set 
if [ $stage -le 1 ]; then
  utils/combine_data.sh data/semisup100k_250k \
    data/train data/train_unsup || exit 1
fi

###############################################################################
# Semi-supervised training using 25k line images supervised data and 
# 22k hours unsupervised data. We use tree, lattices 
# and seed chain system from the previous stage.
###############################################################################
if [ $stage -le 2 ]; then
  local/semisup/chain/run_cnn_chainali_semisupervised_1a.sh \
    --supervised-set train \
    --unsupervised-set train_unsup \
    --sup-chain-dir $exp_root/chain/cnn_e2eali_1b \
    --sup-lat-dir $exp_root/chain/e2e_train_lats \
    --sup-tree-dir $exp_root/chain/tree_e2e \
    --chain-affix "" \
    --tdnn-affix _semisup_1a \
    --exp-root $exp_root || exit 1
fi
