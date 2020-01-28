#!/usr/bin/env bash

# 4d2 is as 4d but adding perturbed training with multiplier=1.0

train_stage=-10
use_gpu=true

. ./cmd.sh
. ./path.sh
. utils/parse_options.sh


if $use_gpu; then
  if ! cuda-compiled; then
    cat <<EOF && exit 1
This script is intended to be used with GPUs but you have not compiled Kaldi with CUDA
If you want to use GPUs (and have them), go to src/, and configure and make on a machine
where "nvcc" is installed.
EOF
  fi
  parallel_opts="--gpu 1"
  num_threads=1
  minibatch_size=512
  dir=exp/nnet4d2_gpu
else
  # Use 4 nnet jobs just like run_4d_gpu.sh so the results should be
  # almost the same, but this may be a little bit slow.
  num_threads=16
  minibatch_size=128
  parallel_opts="--num-threads $num_threads"
  dir=exp/nnet4d2
fi



if [ ! -f $dir/final.mdl ]; then
  steps/nnet2/train_pnorm_fast.sh --stage $train_stage \
     --target-multiplier 1.0 \
     --num-threads "$num_threads" \
     --minibatch-size "$minibatch_size" \
     --parallel-opts "$parallel_opts" \
     --num-jobs-nnet 4 \
     --num-epochs-extra 10 --add-layers-period 1 \
     --num-hidden-layers 2 \
     --mix-up 4000 \
     --initial-learning-rate 0.02 --final-learning-rate 0.004 \
     --cmd "$decode_cmd" \
     --pnorm-input-dim 1000 \
     --pnorm-output-dim 200 \
     data/train data/lang exp/tri3b_ali $dir  || exit 1;
fi

steps/nnet2/decode.sh --config conf/decode.config --cmd "$decode_cmd" --nj 20 \
  --transform-dir exp/tri3b/decode \
  exp/tri3b/graph data/test $dir/decode  &

steps/nnet2/decode.sh --config conf/decode.config --cmd "$decode_cmd" --nj 20 \
  --transform-dir exp/tri3b/decode_ug \
  exp/tri3b/graph_ug data/test $dir/decode_ug

wait

