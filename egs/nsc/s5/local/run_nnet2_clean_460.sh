#!/usr/bin/env bash

. utils/parse_options.sh
. ./cmd.sh

# This shows what you can potentially run; you'd probably want to pick and choose.

use_gpu=true

if $use_gpu; then
  local/nnet2/run_5c_clean_460.sh --use-gpu true # this is on top of fMLLR features.
else
  local/nnet2/run_5c_clean_460.sh --use-gpu false # this is on top of fMLLR features.
fi

