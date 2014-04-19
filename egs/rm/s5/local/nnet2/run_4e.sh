#!/bin/bash

. cmd.sh


# This example runs on top of "raw-fMLLR" features.
# (after reducing splice-width from 7 to 5)
# p-norm version, c.f. 4a.sh which is tanh.

(  steps/nnet2/train_pnorm.sh --splice-width 5 --num-epochs 20 \
     --num-jobs-nnet 4 \
     --initial-learning-rate 0.02 --final-learning-rate 0.004 \
     --num-hidden-layers 2 \
     --num-epochs-extra 10 --add-layers-period 1 \
     --mix-up 4000 \
     --cmd "$decode_cmd" \
     --pnorm-input-dim 1000 \
     --pnorm-output-dim 200 \
     data/train data/lang exp/tri3c_ali exp/nnet4e  || exit 1

   steps/nnet2/decode.sh --config conf/decode.config --cmd "$decode_cmd" --nj 20 --feat-type raw \
     --transform-dir exp/tri3c/decode \
     exp/tri3c/graph data/test exp/nnet4e/decode

   steps/nnet2/decode.sh --config conf/decode.config --cmd "$decode_cmd" --nj 20 --feat-type raw \
     --transform-dir exp/tri3c/decode_ug \
     exp/tri3c/graph_ug data/test exp/nnet4e/decode_ug
 ) 

