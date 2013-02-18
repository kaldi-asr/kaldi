#!/bin/bash

. ./cmd.sh

( # I'm using basically the same setup as for Switchboard 100 hours,
  # but slightly fewer parameters (8M -> 7M) as we have slightly less
  # data (81 hours).
 steps/train_nnet_cpu.sh \
   --mix-up 8000 \
   --initial-learning-rate 0.01 --final-learning-rate 0.001 \
   --num-jobs-nnet 16 --num-hidden-layers 4 \
   --num-parameters 7000000 \
   --cmd "$decode_cmd" \
    data/train_si284 data/lang exp/tri4b_ali_si284 exp/nnet5c1 || exit 1
  
  steps/decode_nnet_cpu.sh --cmd "$decode_cmd" --nj 10 \
    --transform-dir exp/tri4b/decode_bd_tgpr_dev93 \
     exp/tri4b/graph_bd_tgpr data/test_dev93 exp/nnet5c1/decode_bd_tgpr_dev93

  steps/decode_nnet_cpu.sh --cmd "$decode_cmd" --nj 8 \
    --transform-dir exp/tri4b/decode_bd_tgpr_dev93 \
     exp/tri4b/graph_bd_tgpr data/test_dev93 exp/nnet5c1/decode_bd_tgpr_dev93
)


(
 steps/train_nnet_cpu_mmi.sh --boost 0.1 --initial-learning-rate 0.001 \
   --minibatch-size 128 --cmd "$decode_cmd" --transform-dir exp/tri4b_ali_si284 \
   data/train data/lang exp/tri5c1_nnet exp/tri5c1_nnet exp/tri5c1_denlats exp/tri5c1_mmi_a
 
 steps/decode_nnet_cpu.sh --cmd "$decode_cmd" --nj 20 \
   --transform-dir exp/tri3b/decode \
   exp/tri3b/graph data/test exp/tri5c1_mmi_a/decode
)&

