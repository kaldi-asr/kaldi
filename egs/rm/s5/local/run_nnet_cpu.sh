#!/bin/bash

. cmd.sh


( steps/train_nnet_cpu.sh  --num-epochs 20 \
     --num-epochs-extra 10 --add-layers-period 1 \
     --mix-up 4000 --num-iters-final 5 \
     --shrink-interval 3 --alpha 4.0 \
     --initial-learning-rate 0.02 --final-learning-rate 0.004 \
     --cmd "$decode_cmd" \
     --num-parameters 1000000 \
      data/train data/lang exp/tri3b_ali exp/tri4a1_nnet 

   steps/decode_nnet_cpu.sh --cmd "$decode_cmd" --nj 20 \
     --transform-dir exp/tri3b/decode \
     exp/tri3b/graph data/test exp/tri4a1_nnet/decode )

# a2 is as a1, but fewer parameters (1M->750K)
(   steps/train_nnet_cpu.sh  --num-epochs 20 \
     --num-epochs-extra 10 --add-layers-period 1 \
     --mix-up 4000 --num-iters-final 5 \
     --shrink-interval 3 --alpha 4.0 \
     --initial-learning-rate 0.02 --final-learning-rate 0.004 \
     --cmd "$decode_cmd" \
     --num-parameters 750000 \
      data/train data/lang exp/tri3b_ali exp/tri4a2_nnet 

   steps/decode_nnet_cpu.sh --cmd "$decode_cmd" --nj 20 \
     --transform-dir exp/tri3b/decode \
     exp/tri3b/graph data/test exp/tri4a2_nnet/decode ) 
