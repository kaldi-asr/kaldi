#!/usr/bin/env bash

# Copyright 2017  Vimal Manohar
#           2019  Yiming Wang
# Apache 2.0

# This script demonstrates semi-supervised training using ~40 hours of
# supervised data and ~320 hours of unsupervised data.

. ./cmd.sh
. ./path.sh 

set -o pipefail
exp_root=exp/semisup

stage=0

. ./utils/parse_options.sh

###############################################################################
# Train seed chain system using ~40 hours supervised data.
# Here we train i-vector extractor on only the supervised set.
###############################################################################

if [ $stage -le 1 ]; then
  local/semisup/chain/run_tdnn.sh \
    --train-set train \
    --nnet3-affix "" \
    --affix 1a --tree-affix "" \
    --gmm tri3 --exp-root $exp_root || exit 1
fi

if [ $stage -le 2 ]; then
    utils/combine_data.sh data/eval1_2_3_segmented data/eval1_segmented data/eval2_segmented data/eval3_segmented || exit 1
fi

###############################################################################
# Semi-supervised training using ~40 hours supervised data and
# 320 hours unsupervised data. We use i-vector extractor, tree, lattices
# and seed chain system from the previous stage.
###############################################################################

if [ $stage -le 3 ]; then
  local/semisup/chain/run_tdnn_semisupervised.sh \
    --supervised-set train \
    --unsupervised-set eval1_2_3_segmented \
    --sup-chain-dir $exp_root/chain/tdnn_1a_sp \
    --sup-lat-dir $exp_root/chain/tri3_train_sp_lats \
    --sup-tree-dir $exp_root/chain/tree_sp \
    --ivector-root-dir exp/nnet3 \
    --affix 1a \
    --exp-root $exp_root || exit 1

  # [for swahili]
  # %WER 35.2 | 9906 59164 | 67.8 18.4 13.8 3.0 35.2 47.1 | exp/semisup/chain/tdnn_semisup_1a/decode_analysis1_segmented/score_10_0.0/analysis1_segmented_hires.ctm.sys
  # %WER 30.8 | 5322 37120 | 71.9 16.4 11.8 2.7 30.8 47.8 | exp/semisup/chain/tdnn_semisup_1a/decode_analysis2_segmented/score_10_0.0/analysis2_segmented_hires.ctm.sys

  # [for tagalog]
  # %WER 40.8 | 10551 87329 | 64.0 21.4 14.6 4.8 40.8 63.9 | exp/semisup/chain/tdnn_semisup_1a/decode_analysis1_segmented/score_10_0.0/analysis1_segmented_hires.ctm.sys
  # %WER 41.1 | 5933 56887 | 63.8 20.4 15.9 4.9 41.1 71.9 | exp/semisup/chain/tdnn_semisup_1a/decode_analysis2_segmented/score_10_0.0/analysis2_segmented_hires.ctm.sys
fi

