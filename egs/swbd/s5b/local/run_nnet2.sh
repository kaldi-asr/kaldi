#!/bin/bash

# This runs on the 100 hour subset.

. cmd.sh

(  # TODO: change 5a to 5a2.
  if [ ! -f exp/nnet5a2/final.mdl ]; then
    steps/nnet2/train_tanh.sh --stage 215 \
      --mix-up 8000 \
      --initial-learning-rate 0.01 --final-learning-rate 0.001 \
      --num-jobs-nnet 16 --num-hidden-layers 4 \
      --hidden-layer-dim 1024 \
      --cmd "$decode_cmd" \
      data/train_100k_nodup data/lang exp/tri4a exp/nnet5a2 || exit 1;
  fi

  for lm_suffix in tg fsh_tgpr; do
    steps/decode_nnet_cpu.sh --cmd "$decode_cmd" --nj 30 \
      --config conf/decode.config --transform-dir exp/tri4a/decode_eval2000_sw1_${lm_suffix} \
      exp/tri4a/graph_sw1_${lm_suffix} data/eval2000 exp/nnet5a2/decode_eval2000_sw1_${lm_suffix} &
  done
)

