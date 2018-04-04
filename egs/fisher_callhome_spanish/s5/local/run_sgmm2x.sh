#!/bin/bash
# Copyright 2014  Gaurav Kumar.   Apache 2.0

# This is as run_sgmm2.sh but excluding the "speaker-dependent weights",
# so not doing the symmetric SGMM.

. ./cmd.sh

## SGMM on top of LDA+MLLT+SAT features.
if [ ! -f exp/ubm6a/final.mdl ]; then
  steps/train_ubm.sh --silence-weight 0.5 --cmd "$train_cmd" 800 data/train data/lang exp/tri5a_ali exp/ubm6a || exit 1;
fi
# Double the number of SAT states : sanjeev
steps/train_sgmm2.sh  --spk-dep-weights false --cmd "$train_cmd" 10000 120000 \
  data/train data/lang exp/tri5a_ali exp/ubm6a/final.ubm exp/sgmm2x_6a || exit 1;

utils/mkgraph.sh data/lang_test exp/sgmm2x_6a exp/sgmm2x_6a/graph || exit 1;

steps/decode_sgmm2.sh --config conf/decode.config --nj 25 --cmd "$decode_cmd" \
  --transform-dir exp/tri5a/decode_dev  exp/sgmm2x_6a/graph data/dev exp/sgmm2x_6a/decode_dev || exit 1;

steps/decode_sgmm2.sh --use-fmllr true --config conf/decode.config --nj 25 --cmd "$decode_cmd" \
  --transform-dir exp/tri5a/decode_dev  exp/sgmm2x_6a/graph data/dev exp/sgmm2x_6a/decode_dev_fmllr || exit 1;

steps/decode_sgmm2.sh --config conf/decode.config --nj 25 --cmd "$decode_cmd" \
  --transform-dir exp/tri5a/decode_test  exp/sgmm2x_6a/graph data/test exp/sgmm2x_6a/decode_test || exit 1;

steps/decode_sgmm2.sh --use-fmllr true --config conf/decode.config --nj 25 --cmd "$decode_cmd" \
  --transform-dir exp/tri5a/decode_test  exp/sgmm2x_6a/graph data/test exp/sgmm2x_6a/decode_test_fmllr || exit 1;

steps/decode_sgmm2.sh --config conf/decode.config --nj 25 --cmd "$decode_cmd" \
  --transform-dir exp/tri5a/decode_dev2  exp/sgmm2x_6a/graph data/dev2 exp/sgmm2x_6a/decode_dev2 || exit 1;

steps/decode_sgmm2.sh --use-fmllr true --config conf/decode.config --nj 25 --cmd "$decode_cmd" \
  --transform-dir exp/tri5a/decode_dev2  exp/sgmm2x_6a/graph data/dev2 exp/sgmm2x_6a/decode_dev2_fmllr || exit 1;

 #  Now we'll align the SGMM system to prepare for discriminative training.
 steps/align_sgmm2.sh --nj 30 --cmd "$train_cmd" --transform-dir exp/tri5a \
    --use-graphs true --use-gselect true data/train data/lang exp/sgmm2x_6a exp/sgmm2x_6a_ali || exit 1;
 steps/make_denlats_sgmm2.sh --nj 30 --sub-split 210 --cmd "$decode_cmd" --transform-dir exp/tri5a \
   data/train data/lang exp/sgmm2x_6a_ali exp/sgmm2x_6a_denlats
 steps/train_mmi_sgmm2.sh --cmd "$decode_cmd" --transform-dir exp/tri5a --boost 0.2 \
   data/train data/lang exp/sgmm2x_6a_ali exp/sgmm2x_6a_denlats exp/sgmm2x_6a_mmi_b0.2

 for iter in 1 2 3 4; do
  steps/decode_sgmm2_rescore.sh --cmd "$decode_cmd" --iter $iter \
    --transform-dir exp/tri5a/decode_test data/lang data/test exp/sgmm2x_6a/decode_test exp/sgmm2x_6a_mmi_b0.2/decode_test_it$iter &
 done

wait
steps/decode_combine.sh data/test data/lang exp/tri1/decode exp/tri2a/decode exp/combine_1_2a/decode || exit 1;
steps/decode_combine.sh data/test data/lang exp/sgmm2x_4a/decode exp/tri3b_mmi/decode exp/combine_sgmm2x_4a_3b/decode || exit 1;
# combining the sgmm run and the best MMI+fMMI run.
steps/decode_combine.sh data/test data/lang exp/sgmm2x_4a/decode exp/tri3b_fmmi_c/decode_it5 exp/combine_sgmm2x_4a_3b_fmmic5/decode || exit 1;

steps/decode_combine.sh data/test data/lang exp/sgmm2x_4a_mmi_b0.2/decode_it4 exp/tri3b_fmmi_c/decode_it5 exp/combine_sgmm2x_4a_mmi_3b_fmmic5/decode || exit 1;

