#!/bin/bash

stage=0

. ./cmd.sh ## You'll want to change cmd.sh to something that will work on your system.
           ## This relates to the queue.
. utils/parse_options.sh  # e.g. this parses the --stage option if supplied.

if [ $stage -le 0 ]; then
  # data preparation
  local/prepare_data.sh
  for x in cifar{10,100}_{train,test}; do
    image/validate_image_dir.sh data/$x
  done
fi


# cifar10 egs preparation
image/nnet3/get_egs.sh --cmd "$train_cmd" data/cifar10_train data/cifar10_test exp/cifar10_egs
# cifar100 egs preparation
image/nnet3/get_egs.sh --cmd "$train_cmd" data/cifar100_train data/cifar100_test exp/cifar100_egs


# prepare a different version of the egs with 2 instead of 3 archives.
# cifar10 egs preparation
image/nnet3/get_egs.sh --egs-per-archive 30000 --cmd "$train_cmd" data/cifar10_train data/cifar10_test exp/cifar10_egs2
image/nnet3/get_egs.sh --egs-per-archive 30000 --cmd "$train_cmd" data/cifar100_train data/cifar100_test exp/cifar100_egs2


image/nnet3/get_egs.sh --preprocess-opts "--subtract-mean=true --compress=true --compression-method=3" --egs-per-archive 30000 --cmd "$train_cmd" data/cifar10_train data/cifar10_test exp/cifar10_egs2m
image/nnet3/get_egs.sh --preprocess-opts "--subtract-mean=true --compress=true --compression-method=3" --egs-per-archive 30000 --cmd "$train_cmd" data/cifar100_train data/cifar100_test exp/cifar100_egs2m


image/nnet3/get_egs.sh --preprocess-opts "--subtract-mean=true --horizontal-padding=4 --vertical-padding=4 --compress=true --compression-method=3" --egs-per-archive 30000 --cmd "$train_cmd" data/cifar10_train data/cifar10_test exp/cifar10_egs2mp4
image/nnet3/get_egs.sh --preprocess-opts "--subtract-mean=true --horizontal-padding=4 --vertical-padding=4 --compress=true --compression-method=3" --egs-per-archive 30000 --cmd "$train_cmd" data/cifar100_train data/cifar100_test exp/cifar100_egs2mp4


# get egs with padding but no mean subtraction.  Using compression-method=7 to get one byte, since the dynamic range
# is still zero-one.
image/nnet3/get_egs.sh --preprocess-opts "--horizontal-padding=4 --vertical-padding=4 --compress=true --compression-method=7" --egs-per-archive 30000 --cmd "$train_cmd" data/cifar10_train data/cifar10_test exp/cifar10_egs2p4
image/nnet3/get_egs.sh --preprocess-opts "--horizontal-padding=4 --vertical-padding=4 --compress=true --compression-method=7" --egs-per-archive 30000 --cmd "$train_cmd" data/cifar100_train data/cifar100_test exp/cifar100_egs2p4


# more padding.
image/nnet3/get_egs.sh --preprocess-opts "--horizontal-padding=8 --vertical-padding=8 --compress=true --compression-method=7" --egs-per-archive 30000 --cmd "$train_cmd" data/cifar10_train data/cifar10_test exp/cifar10_egs2p8
image/nnet3/get_egs.sh --preprocess-opts "--horizontal-padding=8 --vertical-padding=8 --compress=true --compression-method=7" --egs-per-archive 30000 --cmd "$train_cmd" data/cifar100_train data/cifar100_test exp/cifar100_egs2p8

# nonlinear processing, log(x+0.3) where 0 <= x <= 1 is the original brightness.
image/nnet3/get_egs.sh --preprocess-opts "--log-offset=0.3 --compress=true --compression-method=3" --egs-per-archive 30000 --cmd "$train_cmd" data/cifar10_train data/cifar10_test exp/cifar10_egs2l3
image/nnet3/get_egs.sh --preprocess-opts "--log-offset=0.3 --compress=true --compression-method=3" --egs-per-archive 30000 --cmd "$train_cmd" data/cifar100_train data/cifar100_test exp/cifar100_egs2l3
