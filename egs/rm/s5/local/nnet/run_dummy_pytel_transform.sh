#!/bin/bash

# Copyright 2015  Brno University of Technology (Author: Karel Vesely)
# Apache 2.0

# This example demonstrates how to add i-vector on DNN input (or any other side-info). 
# A fixed vector is pasted to all the frames of an utterance and forwarded to nn-input `as-is', 
# bypassing both the feaure transform and global CMVN normalization.
#
# The i-vector is simulated by a dummy vector [ 0 0 0 ],
# note that all the scripts get an extra option '--ivector'
#
# First we train NN with w/o RBM pre-training, then we do the full recipe:
# RBM pre-training, per-frame training, and sequence-discriminative training.

# Note: With DNNs in RM, the optimal LMWT is 2-6. Don't be tempted to try acwt's like 0.2, 
# the value 0.1 is better both for decoding and sMBR.

. ./cmd.sh ## You'll want to change cmd.sh to something that will work on your system.
           ## This relates to the queue.

. ./path.sh ## Source the tools/utils (import the queue.pl)

dev=data-fbank/test
train=data-fbank/train

dev_original=data/test
train_original=data/train

gmm=exp/tri3b

stage=0
. utils/parse_options.sh

set -uexo pipefail

# Make the FBANK features
[ ! -e $dev ] && if [ $stage -le 0 ]; then
  # Dev set
  utils/copy_data_dir.sh $dev_original $dev rm $dev/{cmvn,feats}.scp
  steps/make_fbank_pitch.sh --nj 10 --cmd "$train_cmd" \
     $dev $dev/log $dev/data
  steps/compute_cmvn_stats.sh $dev $dev/log $dev/data
  # Training set
  utils/copy_data_dir.sh $train_original $train rm $train/{cmvn,feats}.scp
  steps/make_fbank_pitch.sh --nj 10 --cmd "$train_cmd --max-jobs-run 10" \
     $train $train/log $train/data
  steps/compute_cmvn_stats.sh $train $train/log $train/data
  # Split the training set
  utils/subset_data_dir_tr_cv.sh --cv-spk-percent 10 $train ${train}_tr90 ${train}_cv10
fi

# 1) Build NN, no pre-training (script test),
if [ $stage -le 1 ]; then
  # Train the DNN optimizing per-frame cross-entropy.
  dir=exp/dnn4h-dummy-pytel-transform
  ali=${gmm}_ali
  # Train
  $cuda_cmd $dir/log/train_nnet.log \
    steps/nnet/train.sh --hid-layers 4 --hid-dim 1024 --learn-rate 0.008 \
    --pytel-transform utils/numpy_io/pytel_transform_example.py \
    --cmvn-opts "--norm-means=true --norm-vars=true" \
    --delta-opts "--delta-order=2" --splice 5 \
    ${train}_tr90 ${train}_cv10 data/lang $ali $ali $dir
  # Decode (reuse HCLG graph)
  steps/nnet/decode.sh --nj 20 --cmd "$decode_cmd" --config conf/decode_dnn.config --acwt 0.1 \
    $gmm/graph $dev $dir/decode
fi

echo Success
exit 0

# Getting results [see RESULTS file]
# for x in exp/*/decode*; do [ -d $x ] && grep WER $x/wer_* | utils/best_wer.sh; done
