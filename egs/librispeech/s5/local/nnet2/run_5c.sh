#!/bin/bash

# This is neural net training on top of adapted 40-dimensional features.
# 

train_stage=-10
use_gpu=true

train_set="train-clean-100"
test_sets="dev-clean dev-other"

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
  dir=exp/nnet5c_gpu_${train_set}
else
  num_threads=16
  parallel_opts="-pe smp $num_threads" 
  dir=exp/nnet5c_${train_set}
  minibatch_size=128
fi

if [ ! -f $dir/final.mdl ]; then
  if [ "$USER" == dpovey ]; then
     # spread the egs over various machines.  will help reduce overload of any
     # one machine.
     utils/create_split_dir.pl /export/b0{1,2,3,4}/dpovey/kaldi-pure/egs/wsj/s5/$dir/egs $dir/egs/storage
  fi
  steps/nnet2/train_tanh_fast.sh --stage $train_stage \
    --num-threads "$num_threads" \
    --parallel-opts "$parallel_opts" \
    --minibatch-size "$minibatch_size" \
    --num-jobs-nnet 8 \
    --samples-per-iter 400000 \
    --mix-up 8000 \
    --initial-learning-rate 0.01 --final-learning-rate 0.001 \
    --num-hidden-layers 4 --hidden-layer-dim 1024 \
    --cmd "$decode_cmd" \
     data/$train_set data/lang exp/tri4b_ali_${train_set} $dir || exit 1
fi

for test in $test_sets; do
  steps/nnet2/decode.sh --nj 20 --cmd "$decode_cmd" \
    --transform-dir exp/tri4b/decode_tgpr_$test \
    exp/tri4b/graph_tgpr data/$test $dir/decode_tgpr_$test || exit 1;
done

wait

exit 0
