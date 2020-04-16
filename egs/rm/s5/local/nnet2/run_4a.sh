#!/usr/bin/env bash

. ./cmd.sh


# This example runs on top of "raw-fMLLR" features.
# reducing splice-width from 7 to 5

(  steps/nnet2/train_tanh.sh --splice-width 5 --num-epochs 20 \
     --cleanup false \
     --initial-learning-rate 0.02 --final-learning-rate 0.004 \
     --num-hidden-layers 2 \
     --num-epochs-extra 10 --add-layers-period 1 \
     --mix-up 4000 \
     --cmd "$decode_cmd" \
     --hidden-layer-dim 375 \
      data/train data/lang exp/tri3c_ali exp/nnet4a  || exit 1

   steps/nnet2/decode.sh --config conf/decode.config --cmd "$decode_cmd" --nj 20 --feat-type raw \
     --transform-dir exp/tri3c/decode \
     exp/tri3c/graph data/test exp/nnet4a/decode

   steps/nnet2/decode.sh --config conf/decode.config --cmd "$decode_cmd" --nj 20 --feat-type raw \
     --transform-dir exp/tri3c/decode_ug \
     exp/tri3c/graph_ug data/test exp/nnet4a/decode_ug
 ) 

