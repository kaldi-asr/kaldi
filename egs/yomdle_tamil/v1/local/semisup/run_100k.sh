#!/bin/bash

# Copyright 2017  Vimal Manohar
# Apache 2.0

# This script demonstrates semi-supervised training using 25k line images of 
# supervised data and 22k line images of unsupervised data.
# We assume the supervised data is in data/train_sup and unsupervised data
# is in data/train_unsup100k_250k. 
# For LM training, we use 5 million lines of tamil text.

. ./cmd.sh
. ./path.sh 
set -o pipefail
exp_root=exp/semisup_100k
stage=10
. utils/parse_options.sh
for f in data/train_sup/utt2spk data/train_unsup100k_250k/utt2spk \
  data/train_sup/text; do
  if [ ! -f $f ]; then
    echo "$0: Could not find $f"
    exit 1
  fi
done
# Prepare semi-supervised train set 
if [ $stage -le 1 ]; then
  utils/combine_data.sh data/semisup100k_250k \
    data/train_sup data/train_unsup100k_250k || exit 1
fi

# Train seed chain system using 25k line images of supervised data.
#if [ $stage -le 2 ]; then
#    steps/nnet3/align.sh --nj $nj --cmd "$cmd" \
#        --scale-opts '--transition-scale=1.0 --acoustic-scale=1.0 --self-loop-scale=1.0' \
#        data/train_sup data/lang exp/chain/e2e_cnn_1a exp/chain/e2e_ali_train
#fi
###############################################################################
# Semi-supervised training using 25k line images supervised data and 
# 22k hours unsupervised data. We use tree, lattices 
# and seed chain system from the previous stage.
###############################################################################
#if [ $stage -le 10 ]; then
#  local/semisup/chain/run_tdnn_100k_semisupervised_1a.sh \
#    --supervised-set train_sup \
#    --unsupervised-set train_unsup100k_250k \
#    --sup-chain-dir $exp_root/chain/cnn_e2eali_1b \
#    --sup-lat-dir $exp_root/chain/e2e_train_lats \
#    --sup-tree-dir $exp_root/chain/tree_e2e \
#    --chain-affix "" \
#    --stage 18 \
#    --tdnn-affix _semisup_1a \
#    --exp-root $exp_root || exit 1
#fi


if [ $stage -le 11 ]; then
  local/semisup/chain/run_tdnn_100k_semisupervised_1a.sh \
    --supervised-set train_sup \
    --unsupervised-set train_unsup100k_250k \
    --sup-chain-dir $exp_root/chain/cnn_e2eali_1b \
    --sup-lat-dir $exp_root/chain/e2e_train_lats \
    --sup-tree-dir $exp_root/chain/tree_e2e \
    --chain-affix "" \
    --stage 6 \
    --tdnn-affix _semisup_1d \
    --exp-root $exp_root || exit 1
fi
