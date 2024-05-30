#!/usr/bin/env bash

# Copyright 2012-2014  Brno University of Technology (Author: Karel Vesely)
# Apache 2.0

# This example shows how to train a simple autoencoder network.
# We use <tanh>, little different training hyperparameters and MSE objective.

. ./path.sh
. ./cmd.sh

set -eu

# Train,
dir=exp/autoencoder
data_fmllr=data-fmllr-tri3b
labels="ark:feat-to-post scp:$data_fmllr/train/feats.scp ark:- |"
$cuda_cmd $dir/log/train_nnet.log \
  steps/nnet/train.sh --hid-layers 2 --hid-dim 200 --learn-rate 0.00001 \
    --labels "$labels" --num-tgt 40 --train-tool "nnet-train-frmshuff --objective-function=mse" \
    --proto-opts "--no-softmax --activation-type=<Tanh> --hid-bias-mean=0.0 --hid-bias-range=1.0 --param-stddev-factor=0.01" \
    $data_fmllr/train_tr90 $data_fmllr/train_cv10 dummy-dir dummy-dir dummy-dir $dir || exit 1;

# Forward the data,
output_dir=data-autoencoded/test
steps/nnet/make_bn_feats.sh --nj 1 --cmd "$train_cmd" --remove-last-components 0 \
  $output_dir $data_fmllr/test $dir $output_dir/{log,data} || exit 1
