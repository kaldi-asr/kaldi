#!/bin/bash


# run_4d3.sh is as run_4d.sh, but using a newer version of the scripts that
# dump the egs with several frames of labels.


train_stage=-10
use_gpu=true
dir=exp/nnet4d3

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
else
  num_threads=16
  minibatch_size=128
  parallel_opts="-pe smp $num_threads" 
fi


if true || [ ! -f $dir/final.mdl ]; then
  steps/nnet2/train_pnorm_simple2.sh --stage $train_stage \
     --samples-per-iter 200000 \
     --num-threads "$num_threads" \
     --minibatch-size "$minibatch_size" \
     --parallel-opts "$parallel_opts" \
     --num-jobs-nnet 4 \
     --num-epochs 13 --add-layers-period 1 \
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

