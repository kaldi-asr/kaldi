#!/bin/bash

# "nnet2" is the new name for what used to be called the "nnet-cpu" code, and this
# script will eventually supersede run_nnet_cpu.sh  [It's Dan's version of neural
# network training].

# We start from tri3c which is "raw-fMLLR" (a model with regular LDA+MLLT, but where
# the fMLLR is done in the space of the original features).

. cmd.sh


 # The first training is with a small hidden-layer-dim and few epochs, just to
 # get a good point to optimize from.
  steps/nnet2/train_tanh.sh --num-epochs 4 --num-epochs-extra 2 --splice-width 7 \
     --cleanup false \
     --num-hidden-layers 3 --hidden-layer-dim 256 --add-layers-period 1 --cmd "$decode_cmd" \
      data/train data/lang exp/tri3c_ali exp/nnet4c1

 steps/nnet2/decode.sh --config conf/decode.config --cmd "$decode_cmd" --nj 20 \
     --transform-dir exp/tri3c/decode \
     exp/tri3c/graph data/test exp/nnet4c1/decode

 steps/nnet2/retrain_tanh.sh --num-epochs 10 --num-epochs-extra 10 \
     --initial-learning-rate 0.08 --final-learning-rate 0.008 \
     --widen 400 --cmd "$decode_cmd" exp/nnet4c1/egs exp/nnet4c1 exp/nnet5c1

 steps/nnet2/decode.sh --config conf/decode.config --cmd "$decode_cmd" --nj 20 \
     --transform-dir exp/tri3c/decode \
     exp/tri3c/graph data/test exp/nnet5c1/decode

 steps/nnet2/retrain_tanh.sh --num-epochs 10 --num-epochs-extra 10 \
     --mix-up 4000 --initial-learning-rate 0.08 --final-learning-rate 0.008 \
     --cmd "$decode_cmd" exp/nnet4c1/egs exp/nnet5c1 exp/nnet6c1

 steps/nnet2/decode.sh --config conf/decode.config --cmd "$decode_cmd" --nj 20 \
     --transform-dir exp/tri3c/decode \
     exp/tri3c/graph data/test exp/nnet6c1/decode

  steps/nnet2/align.sh --transform-dir exp/tri3c --nj 8 \
     --cmd "$decode_cmd" \
     data/train data/lang exp/nnet6c1 exp/nnet6c1_ali

  steps/nnet2/get_egs.sh  --cmd "$decode_cmd"  --splice-width 7 \
     --transform-dir exp/tri3c/ \
     data/train data/lang exp/nnet6c1_ali exp/nnet6c1_realigned_egs

  steps/nnet2/retrain_tanh.sh --num-epochs 5 --num-epochs-extra 10 \
     --initial-learning-rate 0.04 --final-learning-rate 0.008 \
     --cmd "$decode_cmd" exp/nnet6c1_realigned_egs/egs exp/nnet6c1 exp/nnet7c1 

  steps/nnet2/decode.sh --config conf/decode.config --cmd "$decode_cmd" --nj 20 \
     --transform-dir exp/tri3c/decode \
     exp/tri3c/graph data/test exp/nnet7c1/decode

  steps/nnet2/decode.sh --config conf/decode.config --cmd "$decode_cmd" --nj 20 \
     --transform-dir exp/tri3c/decode_ug \
     exp/tri3c/graph_ug data/test exp/nnet7c1/decode_ug


exit 0;

# using conf/decode.config as we need much larger beams for RM.
steps/make_denlats_nnet_cpu.sh --nj 8 \
   --config conf/decode.config --transform-dir exp/tri3b_ali \
   data/train data/lang exp/tri4a1_nnet exp/tri4a1_denlats

steps/train_nnet_cpu_mmi.sh --cmd "$decode_cmd" --transform-dir exp/tri3b_ali \
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
