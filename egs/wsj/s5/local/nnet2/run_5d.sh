#!/usr/bin/env bash

# This is pnorm neural net training on top of adapted 40-dimensional features.


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
  dir=exp/nnet5d_gpu
else
  # Use 4 nnet jobs just like run_4d_gpu.sh so the results should be
  # almost the same, but this may be a little bit slow.
  num_threads=16
  parallel_opts="--num-threads $num_threads"
  minibatch_size=128
  dir=exp/nnet5d
fi

. ./cmd.sh
. utils/parse_options.sh

if [ ! -f $dir/final.mdl ]; then
  if [ "$USER" == dpovey ]; then
     # spread the egs over various machines.  will help reduce overload of any
     # one machine.
     utils/create_split_dir.pl /export/b0{1,2,3,4}/dpovey/kaldi-pure/egs/wsj/s5/$dir/egs $dir/egs/storage
  fi

  steps/nnet2/train_pnorm_fast.sh --stage $train_stage \
   --samples-per-iter 400000 \
   --parallel-opts "$parallel_opts" \
   --num-threads "$num_threads" \
   --minibatch-size "$minibatch_size" \
   --num-jobs-nnet 8  --mix-up 8000 \
   --initial-learning-rate 0.02 --final-learning-rate 0.004 \
   --num-hidden-layers 4 \
   --pnorm-input-dim 2000 --pnorm-output-dim 400 \
   --cmd "$decode_cmd" \
    data/train_si284 data/lang exp/tri4b_ali_si284 $dir || exit 1
fi


steps/nnet2/decode.sh --cmd "$decode_cmd" --nj 10 \
   --transform-dir exp/tri4b/decode_tgpr_dev93 \
    exp/tri4b/graph_tgpr data/test_dev93 $dir/decode_tgpr_dev93 &

steps/nnet2/decode.sh --cmd "$decode_cmd" --nj 8 \
  --transform-dir exp/tri4b/decode_tgpr_eval92 \
    exp/tri4b/graph_tgpr data/test_eval92 $dir/decode_tgpr_eval92 &

steps/nnet2/decode.sh --cmd "$decode_cmd" --nj 10 \
   --transform-dir exp/tri4b/decode_bd_tgpr_dev93 \
    exp/tri4b/graph_bd_tgpr data/test_dev93 $dir/decode_bd_tgpr_dev93 &

steps/nnet2/decode.sh --cmd "$decode_cmd" --nj 8 \
  --transform-dir exp/tri4b/decode_bd_tgpr_eval92 \
    exp/tri4b/graph_bd_tgpr data/test_eval92 $dir/decode_bd_tgpr_eval92


wait;

