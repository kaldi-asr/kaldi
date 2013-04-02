#!/bin/bash

# This script is invoked from ../run.sh
# It contains some SGMM-related scripts that I am breaking out of the main run.sh for clarity.

. ./cmd.sh ## You'll want to change cmd.sh to something that will work on your system.
           ## This relates to the queue.
[ -f path.sh ] && . ./path.sh



# Note: you might want to try to give the option --spk-dep-weights=false to train_sgmm2.sh;
# this takes out the "symmetric SGMM" part which is not always helpful.

# SGMM system on train data [sgmm4a].  Note: the system we aligned from used the train data for training, but this shouldn't have much effect.


  steps/align_fmllr.sh --nj 30 --cmd "$train_cmd" \
    data/train data/lang exp/tri3b exp/tri3b_ali_train || exit 1;

  steps/train_ubm.sh --cmd "$train_cmd" \
    400 data/train data/lang exp/tri3b_ali_train exp/ubm4a || exit 1;

  steps/train_sgmm2.sh --cmd "$train_cmd" \
    7000 9000 data/train data/lang exp/tri3b_ali_train \
    exp/ubm4a/final.ubm exp/sgmm2_4a || exit 1;

    utils/mkgraph.sh data/lang_test_bg exp/sgmm2_4a exp/sgmm2_4a/graph_bg
    steps/decode_sgmm2.sh --nj 30 --cmd "$decode_cmd" --transform-dir exp/tri3b/decode_bg_test \
      exp/sgmm2_4a/graph_bg data/test exp/sgmm2_4a/decode_bg_test


  steps/align_sgmm2.sh --nj 30 --cmd "$train_cmd" --transform-dir exp/tri3b_ali_train \
    --use-graphs true --use-gselect true data/train data/lang exp/sgmm2_4a exp/sgmm2_4a_ali_train || exit 1;
  steps/make_denlats_sgmm2.sh --nj 30 --sub-split 30 --cmd "$decode_cmd" --transform-dir exp/tri3b_ali_train \
    data/train data/lang exp/sgmm2_4a_ali_train exp/sgmm2_4a_denlats_train

  steps/train_mmi_sgmm2.sh --cmd "$decode_cmd" --transform-dir exp/tri3b_ali_train --boost 0.1 \
    data/train data/lang exp/sgmm2_4a_ali_train exp/sgmm2_4a_denlats_train exp/sgmm2_4a_mmi_b0.1

  for iter in 1 2 3 4; do
    for test in "test"; do # dev93
      steps/decode_sgmm2_rescore.sh --cmd "$decode_cmd" --iter $iter \
        --transform-dir exp/tri3b/decode_bg_${test} data/lang_test_bg data/${test} exp/sgmm2_4a/decode_bg_${test} exp/sgmm2_4a_mmi_b0.1/decode_bg_${test}_it$iter
     done
  done

#  steps/train_mmi_sgmm2.sh --cmd "$decode_cmd" --transform-dir exp/tri3b_ali_train --boost 0.1 \
#   --update-opts "--cov-min-value=0.9" data/train data/lang exp/sgmm2_4a_ali_train exp/sgmm2_4a_denlats_train exp/sgmm2_4a_mmi_b0.1_m0.9

  steps/train_mmi_sgmm2.sh --cmd "$decode_cmd" --transform-dir exp/tri3b_ali_train --boost 0.1 \
    --zero-if-disjoint true data/train data/lang exp/sgmm2_4a_ali_train exp/sgmm2_4a_denlats_train exp/sgmm2_4a_mmi_b0.1_z

  for iter in 1 2 3 4; do
    for test in "test"; do #dev93
      steps/decode_sgmm2_rescore.sh --cmd "$decode_cmd" --iter $iter \
        --transform-dir exp/tri3b/decode_bg_${test} data/lang_test_bg data/${test} exp/sgmm2_4a/decode_bg_${test} \
        exp/sgmm2_4a_mmi_b0.1_z/decode_bg_${test}_it$iter 
     done
  done
 
# Examples of combining some of the best decodings: SGMM+MMI with
# MMI+fMMI on a conventional system.
 
local/score_combine.sh data/test \
   data/lang_test_bg \
   exp/tri3b_fmmi_a/decode_bg_test_it1 \
   exp/sgmm2_4a_mmi_b0.1/decode_bg_test_it1 \
   exp/combine_tri3b_fmmi_a_sgmm2_4a_mmi_b0.1/decode_bg_test_it1_1


# Checking MBR decode of baseline:
cp -r -T exp/sgmm2_4a_mmi_b0.1/decode_bg_test_it3{,.mbr}
local/score_mbr.sh data/test data/lang_test_bg exp/sgmm2_4a_mmi_b0.1/decode_bg_test_it3.mbr
