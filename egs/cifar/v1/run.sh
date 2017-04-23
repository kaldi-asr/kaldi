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
