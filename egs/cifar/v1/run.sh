#!/bin/bash

stage=0
ivector_dimension=30
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

if [ $stage -le 1 ]; then
  #ivector generation 
  local/run_ivector.sh --ivector_dim $ivector_dimension
fi

if [ $stage -le 2 ]; then
  # cifar10 egs preparation
  image/nnet3/get_egs.sh --cmd "$train_cmd" data/cifar10_train data/cifar10_test \
    exp/cifar10_egs exp/ivectors_cifar10_train exp/ivectors_cifar10_test

  echo $ivector_dimension > exp/cifar10_egs/info/ivector_dim

  # cifar100 egs preparation
  image/nnet3/get_egs.sh --cmd "$train_cmd" data/cifar100_train data/cifar100_test \
    exp/cifar100_egs exp/ivectors_cifar100_train exp/ivectors_cifar100_test

  echo $ivector_dimension > exp/cifar100_egs/info/ivector_dim

  # prepare a different version of the egs with 2 instead of 3 archives.
  image/nnet3/get_egs.sh --egs-per-archive 30000 --cmd "$train_cmd" \
    data/cifar10_train data/cifar10_test exp/cifar10_egs2 \
    exp/ivectors_cifar10_train exp/ivectors_cifar10_test
  
  echo $ivector_dimension > exp/cifar10_egs2/info/ivector_dim

  image/nnet3/get_egs.sh --egs-per-archive 30000 --cmd "$train_cmd" \
    data/cifar100_train data/cifar100_test exp/cifar100_egs2 \
    exp/ivectors_cifar100_train exp/ivectors_cifar100_test

  echo $ivector_dimension > exp/cifar100_egs2/info/ivector_dim
fi
