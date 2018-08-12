#!/bin/bash

# Copyright 2017  Vimal Manohar
#           2018  Ashish Arora
# Apache 2.0

# This script demonstrates semi-supervised training using 25k line images of 
# supervised data and 22k line images of unsupervised data.
# We assume the supervised data is in data/train and unsupervised data
# is in data/train_unsup. 
# For LM training, we use 5 million lines of tamil text.

. ./cmd.sh
. ./path.sh 
set -o pipefail
exp_root=exp/semisup_100k
stage=10
. utils/parse_options.sh
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
