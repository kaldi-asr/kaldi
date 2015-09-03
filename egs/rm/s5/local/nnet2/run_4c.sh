#!/bin/bash


# This is neural net training on top of adapted 40-dimensional features.
# The same script works for GPUs, and for CPU only (with --use-gpu false).

train_stage=-10
use_gpu=true

. cmd.sh
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
  parallel_opts="-l gpu=1" 
  num_threads=1
  minibatch_size=512
  dir=exp/nnet4c_gpu
else
  num_threads=16
  minibatch_size=128
  parallel_opts="-pe smp $num_threads" 
  dir=exp/nnet4c
fi



if [ ! -f $dir/final.mdl ]; then
 steps/nnet2/train_tanh_fast.sh --stage $train_stage \
     --num-jobs-nnet 4 \
     --num-threads "$num_threads" \
     --minibatch-size "$minibatch_size" \
     --parallel-opts "$parallel_opts" \
     --num-epochs 20 \
     --add-layers-period 1 \
     --num-hidden-layers 2 \
     --mix-up 4000 \
     --initial-learning-rate 0.02 --final-learning-rate 0.004 \
     --cmd "$decode_cmd" \
     --hidden-layer-dim 375 \
     data/train data/lang exp/tri3b_ali $dir || exit 1;
fi

steps/nnet2/decode.sh --config conf/decode.config --cmd "$decode_cmd" --nj 20 \
  --transform-dir exp/tri3b/decode \
  exp/tri3b/graph data/test $dir/decode  &

steps/nnet2/decode.sh --config conf/decode.config --cmd "$decode_cmd" --nj 20 \
  --transform-dir exp/tri3b/decode_ug \
  exp/tri3b/graph_ug data/test $dir/decode_ug

wait
