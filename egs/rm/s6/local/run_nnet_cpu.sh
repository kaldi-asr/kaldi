#!/bin/bash

. cmd.sh

# result: exp/tri4a1_nnet/decode/wer_2:%WER 1.69 [ 212 / 12533, 26 ins, 44 del, 142 sub ]
 steps/train_nnet_cpu.sh  --num-epochs 20 \
     --num-epochs-extra 10 --add-layers-period 1 \
     --mix-up 4000 --num-iters-final 5 --shrink-interval 3 \
     --num-threads 8 --parallel-opts "-pe smp 8" \
     --initial-learning-rate 0.02 --final-learning-rate 0.004 \
     --cmd "$decode_cmd" \
     --num-parameters 750000 \
      data/train data/lang exp/tri3b_ali exp/tri4a1_nnet  || exit 1

   steps/decode_nnet_cpu.sh --config conf/decode.config --cmd "$decode_cmd" --nj 20 \
     --transform-dir exp/tri3b/decode \
     exp/tri3b/graph data/test exp/tri4a1_nnet/decode || exit 1;

# Don't bother running the MMI training below, it doesn't
# really help.
exit 0;

# using conf/decode.config as we need much larger beams for RM.
steps/make_denlats_nnet_cpu.sh --nj 8 \
   --config conf/decode.config --transform-dir exp/tri3b_ali \
   data/train data/lang exp/tri4a1_nnet exp/tri4a1_denlats

steps/train_nnet_cpu_mmi.sh --cmd "$decode_cmd" --transform-dir exp/tri3b_ali \
   --num-threads 8 \
   data/train data/lang exp/tri4a1_nnet exp/tri4a1_nnet exp/tri4a1_denlats exp/tri4a1_mmi_a

   steps/decode_nnet_cpu.sh --cmd "$decode_cmd" --nj 20 \
     --transform-dir exp/tri3b/decode \
     exp/tri3b/graph data/test exp/tri4a1_mmi_a/decode

(
 steps/train_nnet_cpu_mmi.sh --initial-learning-rate 0.0005 \
   --minibatch-size 128 --cmd "$decode_cmd" --transform-dir exp/tri3b_ali \
   data/train data/lang exp/tri4a1_nnet exp/tri4a1_nnet exp/tri4a1_denlats exp/tri4a1_mmi_b

 steps/decode_nnet_cpu.sh --cmd "$decode_cmd" --nj 20 \
   --transform-dir exp/tri3b/decode \
   exp/tri3b/graph data/test exp/tri4a1_mmi_b/decode
)&


# Get WER on training data before MMI.
 steps/decode_nnet_cpu.sh --cmd "$decode_cmd" --nj 8 \
   --config conf/decode.config --transform-dir exp/tri3b \
   exp/tri3b/graph data/train exp/tri4a1_nnet/decode_train

 # WER on tri3b as baseline, want to see how it compares to tri3b_mmi
 steps/decode.sh --cmd "$decode_cmd" --nj 8 \
   --config conf/decode.config --transform-dir exp/tri3b \
   exp/tri3b/graph data/train exp/tri3b/decode_train
 steps/decode.sh --cmd "$decode_cmd" --nj 8 \
   --config conf/decode.config --transform-dir exp/tri3b \
   exp/tri3b/graph data/train exp/tri3b_mmi/decode_train

(
 steps/train_nnet_cpu_mmi.sh --boost 0.1 --initial-learning-rate 0.0005 \
   --minibatch-size 128 --cmd "$decode_cmd" --transform-dir exp/tri3b_ali \
   data/train data/lang exp/tri4a1_nnet exp/tri4a1_nnet exp/tri4a1_denlats exp/tri4a1_mmi_c

 steps/decode_nnet_cpu.sh --cmd "$decode_cmd" --nj 20 \
   --transform-dir exp/tri3b/decode \
   exp/tri3b/graph data/test exp/tri4a1_mmi_c/decode

 # WER on trainnig data
 steps/decode_nnet_cpu.sh --cmd "$decode_cmd" --nj 8 \
   --config conf/decode.config --transform-dir exp/tri3b \
   exp/tri3b/graph data/train exp/tri4a1_mmi_c/decode_train
)&

(
 steps/train_nnet_cpu_mmi.sh --E 0.5 --boost 0.1 --initial-learning-rate 0.0005 \
   --minibatch-size 128 --cmd "$decode_cmd" --transform-dir exp/tri3b_ali \
   data/train data/lang exp/tri4a1_nnet exp/tri4a1_nnet exp/tri4a1_denlats exp/tri4a1_mmi_d

 steps/decode_nnet_cpu.sh --cmd "$decode_cmd" --nj 20 \
   --transform-dir exp/tri3b/decode \
   exp/tri3b/graph data/test exp/tri4a1_mmi_d/decode

 # WER on trainnig data
 steps/decode_nnet_cpu.sh --cmd "$decode_cmd" --nj 8 \
   --config conf/decode.config --transform-dir exp/tri3b \
   exp/tri3b/graph data/train exp/tri4a1_mmi_d/decode_train
)&

(
 steps/train_nnet_cpu_mmi.sh --E 0.5 --boost 0.1 --initial-learning-rate 0.001 \
   --minibatch-size 128 --cmd "$decode_cmd" --transform-dir exp/tri3b_ali \
   data/train data/lang exp/tri4a1_nnet exp/tri4a1_nnet exp/tri4a1_denlats exp/tri4a1_mmi_e

 steps/decode_nnet_cpu.sh --cmd "$decode_cmd" --nj 20 \
   --transform-dir exp/tri3b/decode \
   exp/tri3b/graph data/test exp/tri4a1_mmi_e/decode

 # WER on trainnig data
 steps/decode_nnet_cpu.sh --cmd "$decode_cmd" --nj 8 \
   --config conf/decode.config --transform-dir exp/tri3b \
   exp/tri3b/graph data/train exp/tri4a1_mmi_e/decode_train
)&

( # _e2 is as _e, but 2 epochs per EBW iter.
 steps/train_nnet_cpu_mmi.sh --epochs-per-ebw-iter 2 --E 0.5 --boost 0.1 --initial-learning-rate 0.001 \
   --minibatch-size 128 --cmd "$decode_cmd" --transform-dir exp/tri3b_ali \
   data/train data/lang exp/tri4a1_nnet exp/tri4a1_nnet exp/tri4a1_denlats exp/tri4a1_mmi_e2

 steps/decode_nnet_cpu.sh --cmd "$decode_cmd" --nj 20 \
   --transform-dir exp/tri3b/decode \
   exp/tri3b/graph data/test exp/tri4a1_mmi_e2/decode

 # WER on trainnig data
 steps/decode_nnet_cpu.sh --cmd "$decode_cmd" --nj 8 \
   --config conf/decode.config --transform-dir exp/tri3b \
   exp/tri3b/graph data/train exp/tri4a1_mmi_e2/decode_train
)&


( # With E = 0.0 it was terrible.  WER is 12.5%
 steps/train_nnet_cpu_mmi.sh --E 0.0 --boost 0.1 --initial-learning-rate 0.001 \
   --minibatch-size 128 --cmd "$decode_cmd" --transform-dir exp/tri3b_ali \
   data/train data/lang exp/tri4a1_nnet exp/tri4a1_nnet exp/tri4a1_denlats exp/tri4a1_mmi_f

 steps/decode_nnet_cpu.sh --cmd "$decode_cmd" --nj 20 \
   --transform-dir exp/tri3b/decode \
   exp/tri3b/graph data/test exp/tri4a1_mmi_f/decode

 # WER on trainnig data
 steps/decode_nnet_cpu.sh --cmd "$decode_cmd" --nj 8 \
   --config conf/decode.config --transform-dir exp/tri3b \
   exp/tri3b/graph data/train exp/tri4a1_mmi_f/decode_train
)&

( 
 steps/train_nnet_cpu_mmi.sh --E 0.25 --boost 0.1 --initial-learning-rate 0.001 \
   --minibatch-size 128 --cmd "$decode_cmd" --transform-dir exp/tri3b_ali \
   data/train data/lang exp/tri4a1_nnet exp/tri4a1_nnet exp/tri4a1_denlats exp/tri4a1_mmi_g

 steps/decode_nnet_cpu.sh --cmd "$decode_cmd" --nj 20 \
   --transform-dir exp/tri3b/decode \
   exp/tri3b/graph data/test exp/tri4a1_mmi_g/decode

 # WER on trainnig data
 steps/decode_nnet_cpu.sh --cmd "$decode_cmd" --nj 8 \
   --config conf/decode.config --transform-dir exp/tri3b \
   exp/tri3b/graph data/train exp/tri4a1_mmi_g/decode_train
)&
