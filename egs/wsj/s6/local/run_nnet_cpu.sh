#!/bin/bash

. ./cmd.sh

 # I'm using basically the same setup as for Switchboard 100 hours,
  # but slightly fewer parameters (8M -> 7M) as we have slightly less
  # data (81 hours).
      #   --num-threads 8 --parallel-opts "-pe smp 8" \
 steps/train_nnet_cpu.sh \
   --mix-up 8000 \
   --initial-learning-rate 0.01 --final-learning-rate 0.001 \
   --num-jobs-nnet 16 --num-hidden-layers 4 \
   --num-parameters 7000000 \
   --cmd "$decode_cmd" \
    data/train_si284 data/lang exp/tri5_ali_si284 exp/nnet6a || exit 1
  
  steps/decode_nnet_cpu.sh --cmd "$decode_cmd" --nj 10 \
    --transform-dir exp/tri5/decode_bd_tgpr_dev93 \
     exp/tri5/graph_bd_tgpr data/test_dev93 exp/nnet6a/decode_bd_tgpr_dev93

  steps/decode_nnet_cpu.sh --cmd "$decode_cmd" --nj 8 \
    --transform-dir exp/tri5/decode_bd_tgpr_dev93 \
     exp/tri5/graph_bd_tgpr data/test_dev93 exp/nnet6a/decode_bd_tgpr_dev93



(
 steps/train_nnet_cpu_mmi.sh --boost 0.1 --initial-learning-rate 0.001 \
   --minibatch-size 128 --cmd "$decode_cmd" --transform-dir exp/tri5_ali_si284 \
   data/train data/lang exp/nnet6a_nnet exp/nnet6a_nnet exp/nnet6a_denlats exp/nnet6a_mmi_a
 
 steps/decode_nnet_cpu.sh --cmd "$decode_cmd" --nj 20 \
   --transform-dir exp/tri3b/decode \
   exp/tri3b/graph data/test exp/nnet6a_mmi_a/decode
)&

