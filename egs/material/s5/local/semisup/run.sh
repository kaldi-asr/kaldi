#!/bin/bash

# Copyright 2017  Vimal Manohar
# Apache 2.0

# This script demonstrates semi-supervised training using 100 hours of 
# supervised data and 250 hours of unsupervised data.
# We assume the supervised data is in data/train_sup and unsupervised data
# is in data/train_unsup100k_250k. 
# For LM training, we only use the supervised set corresponding to 100 hours as 
# opposed to the case in run_50k.sh, where we included part of the 
# transcripts in data/train/text.
# This uses only 100 hours supervised set for i-vector extractor training, 
# which is different from run_50k.sh, which uses combined supervised + 
# unsupervised set.

. ./cmd.sh
. ./path.sh 

set -o pipefail
exp_root=exp/semisup

stage=0

. ./utils/parse_options.sh


###############################################################################
# Prepare lang directories with UNK modeled using phone LM
###############################################################################

if [ $stage -le 1 ]; then
  local/run_unk_model.sh || exit 1

  for lang_dir in data/lang_combined_test; do
    rm -r ${lang_dir}_unk 2>/dev/null || true
    cp -rT data/lang_combined_unk ${lang_dir}_unk
    cp ${lang_dir}/G.fst ${lang_dir}_unk/G.fst
  done
fi

exit 0

###############################################################################
# Train seed chain system using 100 hours supervised data.
# Here we train i-vector extractor on only the supervised set.
###############################################################################

if [ $stage -le 2 ]; then
  local/semisup/chain/run_tdnn.sh \
    --train-set train \
    --nnet3-affix "" \
    --affix 1a --tree-affix "" \
    --gmm tri3 --exp-root $exp_root || exit 1

  # WER on dev                19.23
  # WER on test               19.01
  # Final train prob          -0.1224
  # Final valid prob          -0.1503
  # Final train prob (xent)   -1.6454
  # Final valid prob (xent)   -1.7107
fi

if [ $stage -le 3 ]; then
    utils/combine_data.sh data/eval1_2_segmented_reseg data/eval1_segmented_reseg data/eval2_segmented_reseg || exit 1
fi
exit 0

###############################################################################
# Semi-supervised training using 100 hours supervised data and 
# 250 hours unsupervised data. We use i-vector extractor, tree, lattices 
# and seed chain system from the previous stage.
###############################################################################

if [ $stage -le 4 ]; then
  local/semisup/chain/run_tdnn_semisupervised.sh \
    --supervised-set train \
    --unsupervised-set eval1_2 \
    --sup-chain-dir $exp_root/chain/tdnn_1a_sp \
    --sup-lat-dir $exp_root/chain/tri3_train_sp_unk_lats \
    --sup-tree-dir $exp_root/chain/tree_sp \
    --ivector-root-dir $exp_root/nnet3 \
    --affix 1a \
    --exp-root $exp_root || exit 1

  # WER on dev                          18.70
  # WER on test                         18.18
  # Final output-0 train prob           -0.1345
  # Final output-0 valid prob           -0.1547
  # Final output-0 train prob (xent)    -1.3683
  # Final output-0 valid prob (xent)    -1.4077
  # Final output-1 train prob           -0.6856
  # Final output-1 valid prob           -0.6815
  # Final output-1 train prob (xent)    -1.1224
  # Final output-1 valid prob (xent)    -1.2218
fi

