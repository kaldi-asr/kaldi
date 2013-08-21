#!/bin/bash

# This is neural net training on top of adapted 40-dimensional features.
# 

. cmd.sh

(  steps/nnet2/train_tanh.sh  --num-epochs 20 \
     --num-epochs-extra 10 --add-layers-period 1 \
     --num-hidden-layers 2 \
     --mix-up 4000 \
     --initial-learning-rate 0.02 --final-learning-rate 0.004 \
     --cmd "$decode_cmd" \
     --hidden-layer-dim 375 \
     data/train data/lang exp/tri3b_ali exp/nnet4c_nnet

   steps/decode_nnet_cpu.sh --config conf/decode.config --cmd "$decode_cmd" --nj 20 \
     --transform-dir exp/tri3b/decode \
     exp/tri3b/graph data/test exp/nnet4c_nnet/decode 

   steps/decode_nnet_cpu.sh --config conf/decode.config --cmd "$decode_cmd" --nj 20 \
     --transform-dir exp/tri3b/decode_ug \
     exp/tri3b/graph_ug data/test exp/nnet4c_nnet/decode_ug

)


