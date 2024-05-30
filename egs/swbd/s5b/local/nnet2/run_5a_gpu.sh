#!/usr/bin/env bash

# This runs on the 100 hour subset.
# e.g. of usage:
# local/nnet2/run_5a_gpu.sh --temp-dir /export/m1-01/dpovey/kaldi-dan2/egs/swbd/s5b

temp_dir=
train_stage=-10

. ./cmd.sh
. ./path.sh
! cuda-compiled && cat <<EOF && exit 1 
This script is intended to be used with GPUs but you have not compiled Kaldi with CUDA 
If you want to use GPUs (and have them), go to src/, and configure and make on a machine
where "nvcc" is installed.
EOF


. utils/parse_options.sh

parallel_opts="--gpu 1"  # This is suitable for the CLSP network, you'll likely have to change it.

( 
  if [ ! -f exp/nnet5a_gpu/final.mdl ]; then
    if [ ! -z "$temp_dir" ] && [ ! -e exp/nnet5a_gpu/egs ]; then
      mkdir -p exp/nnet5a_gpu
      mkdir -p $temp_dir/nnet5a_gpu/egs
      ln -s $temp_dir/nnet5a_gpu/egs exp/nnet5a_gpu/
    fi

    steps/nnet2/train_tanh.sh --stage $train_stage \
      --num-jobs-nnet 8 --num-threads 1 --max-change 40.0 \
      --minibatch-size 512 --parallel-opts "$parallel_opts" \
      --mix-up 8000 \
      --initial-learning-rate 0.01 --final-learning-rate 0.001 \
      --num-hidden-layers 4 \
      --hidden-layer-dim 1024 \
      --cmd "$decode_cmd" \
      data/train_100k_nodup data/lang exp/tri4a exp/nnet5a_gpu || exit 1;
  fi

  for lm_suffix in tg fsh_tgpr; do
    steps/nnet2/decode.sh --cmd "$decode_cmd" --nj 30 \
      --config conf/decode.config --transform-dir exp/tri4a/decode_eval2000_sw1_${lm_suffix} \
      exp/tri4a/graph_sw1_${lm_suffix} data/eval2000 exp/nnet5a_gpu/decode_eval2000_sw1_${lm_suffix} &
  done
)

