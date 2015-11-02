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

# Create ark with dummy-ivectors,
[ ! -e data/dummy_ivec.ark ] && cat {$train,$dev}/feats.scp | awk '{ print $1, "[ 0 0 0 ]"; }' >data/dummy_ivec.ark
ivector=ark:data/dummy_ivec.ark

# 1) Build NN, no pre-training (script test),
if [ $stage -le 1 ]; then
  # Train the DNN optimizing per-frame cross-entropy.
  dir=exp/dnn4h-dummy-ivec
  ali=${gmm}_ali
  # Train
  $cuda_cmd $dir/log/train_nnet.log \
    steps/nnet/train.sh --hid-layers 4 --hid-dim 1024 --learn-rate 0.008 \
    --ivector $ivector \
    --cmvn-opts "--norm-means=true --norm-vars=true" \
    --delta-opts "--delta-order=2" --splice 5 \
    ${train}_tr90 ${train}_cv10 data/lang $ali $ali $dir
  # Decode (reuse HCLG graph)
  steps/nnet/decode.sh --nj 20 --cmd "$decode_cmd" --config conf/decode_dnn.config --acwt 0.1 \
    --ivector $ivector \
    $gmm/graph $dev $dir/decode
fi

# 2) Build NN, with pre-training (script test),
if [ $stage -le 2 ]; then
  # Pre-train DBN, i.e. a stack of RBMs (small database, smaller DNN)
  dir=exp/dnn4h-dummy-ivec_pretrain-dbn
  $cuda_cmd $dir/log/pretrain_dbn.log \
    steps/nnet/pretrain_dbn.sh \
      --ivector $ivector \
      --cmvn-opts "--norm-means=true --norm-vars=true" \
      --delta-opts "--delta-order=2" --splice 5 \
      --hid-dim 1024 --rbm-iter 20 $train $dir
fi

if [ $stage -le 3 ]; then
  # Train the DNN optimizing per-frame cross-entropy.
  dir=exp/dnn4h-dummy-ivec_pretrain-dbn_dnn
  ali=${gmm}_ali
  feature_transform=exp/dnn4h-dummy-ivec_pretrain-dbn/final.feature_transform
  dbn=exp/dnn4h-dummy-ivec_pretrain-dbn/6.dbn
  # Train
  $cuda_cmd $dir/log/train_nnet.log \
    steps/nnet/train.sh --feature-transform $feature_transform --dbn $dbn --hid-layers 0 --learn-rate 0.008 \
    --ivector $ivector \
    ${train}_tr90 ${train}_cv10 data/lang $ali $ali $dir
  # Decode (reuse HCLG graph)
  steps/nnet/decode.sh --nj 20 --cmd "$decode_cmd" --config conf/decode_dnn.config --acwt 0.1 \
    --ivector $ivector \
    $gmm/graph $dev $dir/decode
fi


# Sequence training using sMBR criterion, we do Stochastic-GD with per-utterance updates.
# Note: With DNNs in RM, the optimal LMWT is 2-6. Don't be tempted to try acwt's like 0.2, 
# the value 0.1 is better both for decoding and sMBR.
dir=exp/dnn4h-dummy-ivec_pretrain-dbn_dnn_smbr
srcdir=exp/dnn4h-dummy-ivec_pretrain-dbn_dnn
acwt=0.1

if [ $stage -le 4 ]; then
  # First we generate lattices and alignments:
  steps/nnet/align.sh --nj 20 --cmd "$train_cmd" \
    --ivector $ivector \
    $train data/lang $srcdir ${srcdir}_ali
  steps/nnet/make_denlats.sh --nj 20 --cmd "$decode_cmd" --config conf/decode_dnn.config --acwt $acwt \
    --ivector $ivector \
    $train data/lang $srcdir ${srcdir}_denlats
fi

if [ $stage -le 5 ]; then
  # Re-train the DNN by 6 iterations of sMBR 
  steps/nnet/train_mpe.sh --cmd "$cuda_cmd" --num-iters 6 --acwt $acwt --do-smbr true \
    --ivector $ivector \
    $train data/lang $srcdir ${srcdir}_ali ${srcdir}_denlats $dir || exit 1
  # Decode
  for ITER in 1 3 6; do
    steps/nnet/decode.sh --nj 20 --cmd "$decode_cmd" --config conf/decode_dnn.config \
      --ivector $ivector \
      --nnet $dir/${ITER}.nnet --acwt $acwt \
      $gmm/graph $dev $dir/decode_it${ITER} || exit 1
  done 
fi

echo Success
exit 0

# Getting results [see RESULTS file]
# for x in exp/*/decode*; do [ -d $x ] && grep WER $x/wer_* | utils/best_wer.sh; done
