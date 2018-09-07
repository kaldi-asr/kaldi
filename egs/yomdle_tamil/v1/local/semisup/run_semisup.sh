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

mkdir -p data/train_unsup/data
if [ $stage -le 0 ]; then
  echo "stage 0: Processing train unsupervised data...$(date)"
  local/semisup/process_data.py data/download/ \
    data/local/splits/train_unsup.txt \
    data/train_unsup
  image/fix_data_dir.sh data/train_unsup
fi

if [ $stage -le 1 ]; then
  echo "stage 1: Obtaining image groups. calling get_image2num_frames..."
  image/get_image2num_frames.py --feat-dim 40 data/train_unsup
  image/get_allowed_lengths.py --frame-subsampling-factor 4 10 data/train_unsup
  echo "Extracting features and calling compute_cmvn_stats: $(date) "
  local/extract_features.sh --nj $nj --cmd "$cmd" --feat-dim 40 data/train_unsup
  steps/compute_cmvn_stats.sh data/train_unsup || exit 1;
  image/fix_data_dir.sh data/train_unsup
fi

for f in data/train/utt2spk data/train_unsup/utt2spk \
  data/train/text; do
  if [ ! -f $f ]; then
    echo "$0: Could not find $f"
    exit 1;
  fi
done

# Prepare semi-supervised train set 
if [ $stage -le 1 ]; then
  utils/combine_data.sh data/semisup100k_250k \
    data/train_aug data/train_unsup || exit 1
fi

###############################################################################
# Semi-supervised training using 25k line images supervised data and 
# 22k hours unsupervised data. We use tree, lattices 
# and seed chain system from the previous stage.
###############################################################################
if [ $stage -le 2 ]; then
  local/semisup/chain/run_cnn_chainali_semisupervised_1b.sh \
    --supervised-set train_aug \
    --unsupervised-set train_unsup \
    --sup-chain-dir exp/chain/cnn_e2eali_1b \
    --sup-lat-dir exp/chain/e2e_train_lats \
    --sup-tree-dir exp/chain/tree_e2e \
    --chain-affix "" \
    --tdnn-affix _semisup_1a \
    --exp-root $exp_root || exit 1
fi
