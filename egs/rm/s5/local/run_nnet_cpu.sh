#!/bin/bash

. cmd.sh


# result: exp/tri4a2_nnet/decode/wer_2:%WER 1.69 [ 212 / 12533, 26 ins, 44 del, 142 sub ]
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

# using conf/decode.config as we need much larger beams for RM.
steps/make_denlats_nnet_cpu.sh --nj 8 \
   --config conf/decode.config --transform-dir exp/tri3b_ali \
   data/train data/lang exp/tri4a2_nnet exp/tri4a2_denlats

steps/train_nnet_cpu_mmi.sh --cmd "$decode_cmd" --transform-dir exp/tri3b_ali \
   data/train data/lang exp/tri4a2_nnet exp/tri4a2_nnet exp/tri4a2_denlats exp/tri4a2_mmi_a

   steps/decode_nnet_cpu.sh --cmd "$decode_cmd" --nj 20 \
     --transform-dir exp/tri3b/decode \
     exp/tri3b/graph data/test exp/tri4a2_mmi_a/decode

(
 steps/train_nnet_cpu_mmi.sh --initial-learning-rate 0.0005 \
   --minibatch-size 128 --cmd "$decode_cmd" --transform-dir exp/tri3b_ali \
   data/train data/lang exp/tri4a2_nnet exp/tri4a2_nnet exp/tri4a2_denlats exp/tri4a2_mmi_b

 steps/decode_nnet_cpu.sh --cmd "$decode_cmd" --nj 20 \
   --transform-dir exp/tri3b/decode \
   exp/tri3b/graph data/test exp/tri4a2_mmi_b/decode
)&

(
 steps/train_nnet_cpu_mmi.sh --boost 0.1 --initial-learning-rate 0.0005 \
   --minibatch-size 128 --cmd "$decode_cmd" --transform-dir exp/tri3b_ali \
   data/train data/lang exp/tri4a2_nnet exp/tri4a2_nnet exp/tri4a2_denlats exp/tri4a2_mmi_c

 steps/decode_nnet_cpu.sh --cmd "$decode_cmd" --nj 20 \
   --transform-dir exp/tri3b/decode \
   exp/tri3b/graph data/test exp/tri4a2_mmi_c/decode
)&

(
 steps/train_nnet_cpu_mmi.sh --boost 0.1 --initial-learning-rate 0.001 \
   --minibatch-size 128 --cmd "$decode_cmd" --transform-dir exp/tri3b_ali \
   data/train data/lang exp/tri4a2_nnet exp/tri4a2_nnet exp/tri4a2_denlats exp/tri4a2_mmi_c

 steps/decode_nnet_cpu.sh --cmd "$decode_cmd" --nj 20 \
   --transform-dir exp/tri3b/decode \
   exp/tri3b/graph data/test exp/tri4a2_mmi_c/decode
)&
