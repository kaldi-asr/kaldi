#!/usr/bin/env bash

# This is neural net training on top of adapted 40-dimensional features.
# 

. ./cmd.sh

uid=$1       # Here $uid is the LM identifier, e.g '3g'
test1=$2
test2=$3


 steps/nnet2/train_tanh.sh \
   --mix-up 8000 \
   --initial-learning-rate 0.01 --final-learning-rate 0.001 \
   --num-hidden-layers 4 --hidden-layer-dim 1024 \
   --cmd "$train_cmd" \
   data/train data/lang_test_4g exp/tri4a_ali exp/nnet5c || exit 1
  
  steps/nnet2/decode.sh --cmd "$decode_cmd" --nj 10 \
    --transform-dir exp/tri4a/decode_4g \
     exp/tri4a/graph_4g data/test exp/nnet5c/decode_4g_test

if [ -d $test2 ]; then
  steps/decode_nnet_cpu.sh --cmd "$decode_cmd" --nj 4 \
    --transform-dir exp/tri4a/decode_${uid}_$test2 \
     exp/tri4a/graph_${uid} data/$test2 exp/nnet5c/decode_${uid}_$test2
fi


