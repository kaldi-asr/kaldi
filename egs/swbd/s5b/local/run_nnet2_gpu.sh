#!/bin/bash

# This runs on the 100 hour subset.  This version of the recipe runs on GPUs.
# We assume you have 8 GPU machines.  You have to use --num-threads 1 so it will
# use the version of the code that can use GPUs.
# We assume the queue is set up as in JHU (or as in the "Kluster" project
# on Sourceforge) where "gpu" is a consumable resource that you can set to
# number of GPU cards a machine has.

. cmd.sh

( 
  if [ ! -f exp/nnet5b/final.mdl ]; then
    steps/nnet2/train_tanh.sh --cmd "$decode_cmd -l gpu=1" --parallel-opts "" --stage 0 \
      --num-threads 1 \
      --mix-up 8000 \
      --initial-learning-rate 0.01 --final-learning-rate 0.001 \
      --num-jobs-nnet 8 --num-hidden-layers 4 \
      --hidden-layer-dim 1024 \
      data/train_100k_nodup data/lang exp/tri4a exp/nnet5b || exit 1;
  fi

  for lm_suffix in tg fsh_tgpr; do
    steps/decode_nnet_cpu.sh --cmd "$decode_cmd" --nj 30 \
      --config conf/decode.config --transform-dir exp/tri4a/decode_eval2000_sw1_${lm_suffix} \
      exp/tri4a/graph_sw1_${lm_suffix} data/eval2000 exp/nnet5b/decode_eval2000_sw1_${lm_suffix} &
  done
)

