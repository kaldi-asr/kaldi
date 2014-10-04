#!/bin/bash

train_set="train-clean-100"
test_sets="dev-clean dev-other"

. utils/parse_options.sh
. ./cmd.sh

# This shows what you can potentially run; you'd probably want to pick and choose.

use_gpu=true

if $use_gpu; then
  local/nnet2/run_5b_gpu.sh # various VTLN combinations,  Mel-filterbank features, si284 train (multiplied by 5).

  # this is on top of fMLLR features.
  local/nnet2/run_5c.sh --use-gpu true --train-set "$train_set" --test-sets "$test_sets"

  local/nnet2/run_6c_gpu.sh # this is discriminative training of tanh neural nets on top of run_5c_gpu.sh
  local/nnet2/run_5d.sh --use-gpu true # this is p-norm training on top of fMLLR features.  <THIS IS THE MAIN RECIPE>
  local/nnet2/run_5e_gpu.sh # this is ensemble training of p-norm nnets on top of fMLLR features.
  local/nnet2/run_6d_gpu.sh # this is discriminative training of p-norm neural nets on top of run_5d_gpu.sh
else
  local/nnet2/run_5b.sh # various VTLN combinations, Mel-filterbank features,  si284 train (multiplied by 5).
  local/nnet2/run_5c.sh --use-gpu false # this is on top of fMLLR features.
  local/nnet2/run_5d.sh --use-gpu false # this is p-norm on top of fMLLR features.  <THIS IS THE MAIN RECIPE>
fi


