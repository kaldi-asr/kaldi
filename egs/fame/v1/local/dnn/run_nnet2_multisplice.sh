#!/usr/bin/env bash
# Copyright 2017  Radboud University (Author: Emre Yilmaz)
#
# This script is based on run_nnet2_multisplice.sh in
# egs/fisher_english/s5/local/online. It has been modified
# for speaker recognition.

stage=1
train_stage=-10
use_gpu=true
set -e
. ./cmd.sh
. ./path.sh

. utils/parse_options.sh

# assume use_gpu=true since it would be way too slow otherwise.

if $use_gpu; then
  if ! cuda-compiled; then
    cat <<EOF && exit 1
This script is intended to be used with GPUs but you have not compiled Kaldi with CUDA
If you want to use GPUs (and have them), go to src/, and configure and make on a machine
where "nvcc" is installed.  Otherwise, call this script with --use-gpu false
EOF
  fi
  parallel_opts="--gpu 1"
  num_threads=1
  minibatch_size=512
else
  # Use 4 nnet jobs just like run_4d_gpu.sh so the results should be
  # almost the same, but this may be a little bit slow.
  num_threads=16
  minibatch_size=128
  parallel_opts="--num-threads $num_threads"
fi

dir=exp/nnet2_online/nnet_ms_a
mkdir -p exp/nnet2_online
decode_cmd='run.pl'
# Stages 1 through 5 are done in run_nnet2_common.sh,
# so it can be shared with other similar scripts.
local/dnn/run_nnet2_common.sh --stage $stage

if [ $stage -le 6 ]; then

  sid/nnet2/train_multisplice_accel2.sh --stage $train_stage \
    --feat-type raw \
    --splice-indexes "layer0/-2:-1:0:1:2 layer1/-1:2 layer3/-3:3 layer4/-7:2" \
    --num-epochs 10 \
    --num-hidden-layers 6 \
    --num-jobs-initial 3 --num-jobs-final 8 \
    --num-threads "$num_threads" \
    --minibatch-size "$minibatch_size" \
    --parallel-opts "$parallel_opts" \
    --mix-up 10500 \
    --initial-effective-lrate 0.0015 --final-effective-lrate 0.00015 \
    --cmd "$decode_cmd" \
    --egs-dir "$common_egs_dir" \
    --pnorm-input-dim 2000 \
    --pnorm-output-dim 200 \
    data/train_hires_asr data/lang exp/tri3 $dir  || exit 1;

fi
