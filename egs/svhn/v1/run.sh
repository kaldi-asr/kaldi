#!/bin/bash

stage=0

. ./cmd.sh ## You'll want to change cmd.sh to something that will work on your system.
           ## This relates to the queue.
. utils/parse_options.sh  # e.g. this parses the --stage option if supplied.

if [ $stage -le 0 ]; then
  # data preparation
  local/prepare_data.sh
  for x in test train extra train_all; do
    image/validate_image_dir.sh data/$x
  done
fi


if [ $stage -le 1 ]; then
  # egs preparation
  image/nnet3/get_egs.sh --egs-per-archive 50000 --cmd "$cmd" data/train_all data/test exp/egs
fi
