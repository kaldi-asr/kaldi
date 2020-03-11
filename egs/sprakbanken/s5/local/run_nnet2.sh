#!/usr/bin/env bash

. ./cmd.sh

# This shows what you can potentially run; you'd probably want to pick and choose.

use_gpu=true

if $use_gpu; then
  local/nnet2/run_5b_gpu.sh # various VTLN combinations,  Mel-filterbank features, si284 train (multiplied by 5).
  local/nnet2/run_5c_gpu.sh # this is on top of fMLLR features.
else
  local/nnet2/run_5b.sh # various VTLN combinations, Mel-filterbank features,  si284 train (multiplied by 5).
  local/nnet2/run_5c.sh # this is on top of fMLLR features.
fi


