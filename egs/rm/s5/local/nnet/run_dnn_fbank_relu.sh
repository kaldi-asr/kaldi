#!/bin/bash

# Copyright 2012-2014  Brno University of Technology (Author: Karel Vesely)
# Apache 2.0

# This example script trains a DNN on top of FBANK features.
# The training is done in 3 stages,
#
# 1) RBM pre-training:
#    in this unsupervised stage we train stack of RBMs,
#    a good starting point for frame cross-entropy trainig.
# 2) frame cross-entropy training:
#    the objective is to classify frames to correct pdfs.
# 3) sequence-training optimizing sMBR:
#    the objective is to emphasize state-sequences with better
#    frame accuracy w.r.t. reference alignment.

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
. utils/parse_options.sh || exit 1;

set -euxo pipefail

# Make the FBANK features
[ ! -e $dev ] && if [ $stage -le 0 ]; then
  # Dev set
  utils/copy_data_dir.sh $dev_original $dev || exit 1; rm $dev/{cmvn,feats}.scp
  steps/make_fbank_pitch.sh --nj 10 --cmd "$train_cmd" \
     $dev $dev/log $dev/data || exit 1;
  steps/compute_cmvn_stats.sh $dev $dev/log $dev/data || exit 1;
  # Training set
  utils/copy_data_dir.sh $train_original $train || exit 1; rm $train/{cmvn,feats}.scp
  steps/make_fbank_pitch.sh --nj 10 --cmd "$train_cmd --max-jobs-run 10" \
     $train $train/log $train/data || exit 1;
  steps/compute_cmvn_stats.sh $train $train/log $train/data || exit 1;
  # Split the training set
  utils/subset_data_dir_tr_cv.sh --cv-spk-percent 10 $train ${train}_tr90 ${train}_cv10
fi

# Tuned with 6x1024 Relu units,
lrate=0.001
param_std=0.02

# Original Relu,
if [ $stage -le 1 ]; then
  # Train the DNN optimizing per-frame cross-entropy.
  dir=exp/dnn4d-6L1024-relu-fbank
  ali=${gmm}_ali
  # Train
  $cuda_cmd $dir/log/train_nnet.log \
    steps/nnet/train.sh --learn-rate $lrate \
    --cmvn-opts "--norm-means=true --norm-vars=true" \
    --delta-opts "--delta-order=2" --splice 5 \
    --hid-layers 6 --hid-dim 1024 \
    --proto-opts "--activation-type <ParametricRelu> --param-stddev-factor $param_std --hid-bias-mean 0 --hid-bias-range 0 --no-glorot-scaled-stddev --no-smaller-input-weights" \
    ${train}_tr90 ${train}_cv10 data/lang $ali $ali $dir
  # Decode (reuse HCLG graph)
  steps/nnet/decode.sh --nj 20 --cmd "$decode_cmd" --config conf/decode_dnn.config --acwt 0.1 \
    $gmm/graph $dev $dir/decode
fi

# Parametric Relu,
lr_alpha=1.0
lr_beta=0.75
if [ $stage -le 2 ]; then
  # Train the DNN optimizing per-frame cross-entropy.
  dir=exp/dnn4d-6L1024-relu-fbank-alpha-beta
  ali=${gmm}_ali
  # Train
  $cuda_cmd $dir/log/train_nnet.log \
    steps/nnet/train.sh --learn-rate $lrate \
    --cmvn-opts "--norm-means=true --norm-vars=true" \
    --delta-opts "--delta-order=2" --splice 5 \
    --hid-layers 6 --hid-dim 1024 \
    --proto-opts "--activation-type=<ParametricRelu> --activation-opts=<AlphaLearnRateCoef>_${lr_alpha}_<BetaLearnRateCoef>_${lr_beta} --param-stddev-factor $param_std --hid-bias-mean 0 --hid-bias-range 0 --no-glorot-scaled-stddev --no-smaller-input-weights" \
    ${train}_tr90 ${train}_cv10 data/lang $ali $ali $dir
  # Decode (reuse HCLG graph)
  steps/nnet/decode.sh --nj 20 --cmd "$decode_cmd" --config conf/decode_dnn.config --acwt 0.1 \
    $gmm/graph $dev $dir/decode
fi


# Sequence training using sMBR criterion, we do Stochastic-GD with per-utterance updates.
# Note: With DNNs in RM, the optimal LMWT is 2-6. Don't be tempted to try acwt's like 0.2,
# the value 0.1 is better both for decoding and sMBR.
dir=exp/dnn4d-6L1024-relu-fbank-alpha-beta_smbr
srcdir=exp/dnn4d-6L1024-relu-fbank-alpha-beta
acwt=0.1

if [ $stage -le 3 ]; then
  # First we generate lattices and alignments:
  steps/nnet/align.sh --nj 20 --cmd "$train_cmd" \
    $train data/lang $srcdir ${srcdir}_ali
  steps/nnet/make_denlats.sh --nj 20 --cmd "$decode_cmd" --config conf/decode_dnn.config --acwt $acwt \
    $train data/lang $srcdir ${srcdir}_denlats
fi

lrate=0.000001 # an 10x smaller than with Sigmoid,
if [ $stage -le 4 ]; then
  # Re-train the DNN by 6 iterations of sMBR
  steps/nnet/train_mpe.sh --cmd "$cuda_cmd" --num-iters 6 --learn-rate $lrate --acwt $acwt --do-smbr true \
    $train data/lang $srcdir ${srcdir}_ali ${srcdir}_denlats $dir
  # Decode
  for ITER in 6 3 1; do
    steps/nnet/decode.sh --nj 20 --cmd "$decode_cmd" --config conf/decode_dnn.config \
      --nnet $dir/${ITER}.nnet --acwt $acwt \
      $gmm/graph $dev $dir/decode_it${ITER}
  done
fi



echo Success
exit 0

# Getting results [see RESULTS file]
# for x in exp/*/decode*; do [ -d $x ] && grep WER $x/wer_* | utils/best_wer.sh; done
