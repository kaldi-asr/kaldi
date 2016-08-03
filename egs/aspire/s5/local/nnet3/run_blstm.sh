#!/bin/bash

# this is a script to train the nnet3 blstm acoustic model
# it is based on blstm used in fisher_swbd recipe

stage=7
affix=bidirectional
train_stage=-10
egs_stage=0
common_egs_dir=
remove_egs=true

. ./cmd.sh
. ./path.sh
. ./utils/parse_options.sh

local/nnet3/run_lstm.sh  --stage $stage --train-stage $train_stage --egs-stage "$egs_stage" \
                         --affix $affix --lstm-delay " [-1,1] [-2,2] [-3,3] " --label-delay 0 \
                         --cell-dim 1024 --recurrent-projection-dim 128 --non-recurrent-projection-dim 128 \
                         --chunk-left-context 40 --chunk-right-context 40 \
                         --extra-left-context 50 --extra-right-context 50 \
                         --common-egs-dir "$common_egs_dir" --remove-egs "$remove_egs"
