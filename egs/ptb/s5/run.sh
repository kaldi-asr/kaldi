#!/usr/bin/env bash

stage=0

. ./cmd.sh ## You'll want to change cmd.sh to something that will work on your system.
           ## This relates to the queue.
. utils/parse_options.sh  # e.g. this parses the --stage option if supplied.


if [ $stage -le 0 ]; then
  # download ptb data
  local/rnnlm/download_ptb.sh || exit 1;
fi
if [ $stage -le 1 ]; then
  # format ptb data
  local/rnnlm/prepare_rnnlm_data.sh || exit 1;
fi

if [ $stage -le 2 ]; then
  local/rnnlm/run_tdnn.sh || exit 1;
fi
