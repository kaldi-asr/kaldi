#!/bin/bash

# This runs on the 100 hour subset.

. cmd.sh

( 
 steps/train_nnet_cpu.sh \
   --mix-up 8000 \
   --initial-learning-rate 0.01 --final-learning-rate 0.001 \
   --num-jobs-nnet 16 --num-hidden-layers 4 \
   --num-parameters 8000000 \
   --cmd "$decode_cmd" \
    data/train_100k_nodup data/lang exp/tri4a exp/nnet5a

  steps/decode_nnet_cpu.sh --cmd "$decode_cmd" --nj 30 \
    --config conf/decode.config --transform-dir exp/tri4a/decode_train_dev \
   exp/tri4a/graph data/train_dev exp/nnet5a/decode_train_dev &

  steps/decode_nnet_cpu.sh --cmd "$decode_cmd" --nj 30 \
    --config conf/decode.config --transform-dir exp/tri4a/decode_eval2000 \
   exp/tri4a/graph data/eval2000 exp/nnet5a/decode_eval2000 &
)

