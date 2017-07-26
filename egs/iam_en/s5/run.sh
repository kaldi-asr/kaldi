#!/bin/bash

stage=0
variance_floor_val=0.1
. ./cmd.sh ## You'll want to change cmd.sh to something that will work on your system.
           ## This relates to the queue.
. utils/parse_options.sh  # e.g. this parses the --stage option if supplied.

if [ $stage -le 0 ]; then
  # data preparation
  local/prepare_data.sh
fi

if [ $stage -le 2 ]; then

  ## Starting basic training on features
  ## passing value for variance floor
  steps/train_mono.sh --nj 30 \
    --variance_floor_val $variance_floor_val data/train data/lang exp/mono
fi
