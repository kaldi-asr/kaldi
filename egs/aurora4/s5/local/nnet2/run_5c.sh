#!/bin/bash

# This is neural net training on top of adapted 40-dimensional features.
# 

. ./cmd.sh

( 
 steps/nnet2/train_tanh.sh \
   --mix-up 8000 \
   --initial-learning-rate 0.01 --final-learning-rate 0.001 \
   --num-hidden-layers 4 --hidden-layer-dim 1024 \
   --cmd "$decode_cmd" \
    data/train_si284 data/lang exp/tri4b_ali_si284 exp/nnet5c || exit 1
  
  steps/nnet2/decode.sh --cmd "$decode_cmd" --nj 10 \
    --transform-dir exp/tri4b/decode_bd_tgpr_dev93 \
     exp/tri4b/graph_bd_tgpr data/test_dev93 exp/nnet5c/decode_bd_tgpr_dev93

  steps/nnet2/decode.sh --cmd "$decode_cmd" --nj 8 \
    --transform-dir exp/tri4b/decode_bd_tgpr_eval92 \
     exp/tri4b/graph_bd_tgpr data/test_eval92 exp/nnet5c/decode_bd_tgpr_eval92
)

