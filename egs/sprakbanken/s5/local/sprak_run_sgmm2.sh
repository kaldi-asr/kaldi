#!/bin/bash

# This script is invoked from ../run.sh
# It contains some SGMM-related scripts that I am breaking out of the main run.sh for clarity.

. cmd.sh

# Note: you might want to try to give the option --spk-dep-weights=false to train_sgmm2.sh;
# this takes out the "symmetric SGMM" part which is not always helpful.


test=$1

if [ ! -d exp/tri4b_ali ]; then
  steps/align_fmllr.sh --nj 30 --cmd "$train_cmd" \
    data/train data/lang exp/tri4b exp/tri4b_ali || exit 1;
fi
  steps/train_ubm.sh --cmd "$train_cmd" \
    400 data/train data/lang exp/tri4b_ali exp/ubm5a || exit 1;

  steps/train_sgmm2.sh --cmd "$train_cmd" \
    7000 9000 data/train data/lang exp/tri4b_ali \
    exp/ubm5a/final.ubm exp/sgmm2_5a || exit 1;

  (
    utils/mkgraph.sh data/lang_test_3g exp/sgmm2_5a exp/sgmm2_5a/graph_3g
    steps/decode_sgmm2.sh --nj 7 --cmd "$decode_cmd" --transform-dir exp/tri4b/decode_3g_${test} \
      exp/sgmm2_5a/graph_3g data/${test} exp/sgmm2_5a/decode_3g_${test}
  ) &

  steps/align_sgmm2.sh --nj 30 --cmd "$train_cmd" --transform-dir exp/tri4b_ali \
    --use-graphs true --use-gselect true data/train data/lang exp/sgmm2_5a exp/sgmm2_5a_ali || exit 1;
  steps/make_denlats_sgmm2.sh --nj 30 --sub-split 2 --cmd "$decode_cmd" --transform-dir exp/tri4b_ali \
    data/train data/lang exp/sgmm2_5a_ali exp/sgmm2_5a_denlats

  wait

  steps/train_mmi_sgmm2.sh --cmd "$decode_cmd" --transform-dir exp/tri4b_ali --boost 0.1 \
    data/train data/lang exp/sgmm2_5a_ali exp/sgmm2_5a_denlats exp/sgmm2_5a_mmi_b0.1

  wait

  for iter in 1 2 3 4; do
    steps/decode_sgmm2_rescore.sh --cmd "$decode_cmd" --iter $iter \
      --transform-dir exp/tri4b/decode_3g_${test} data/lang_test_3g data/${test} exp/sgmm2_5a/decode_3g_${test} \
      exp/sgmm2_5a_mmi_b0.1/decode_3g_${test}_it$iter &
  done

  wait

  steps/train_mmi_sgmm2.sh --cmd "$decode_cmd" --transform-dir exp/tri4b_ali --boost 0.1 \
   --update-opts "--cov-min-value=0.9" data/train data/lang exp/sgmm2_5a_ali exp/sgmm2_5a_denlats exp/sgmm2_5a_mmi_b0.1_m0.9

   wait

  for iter in 1 2 3 4; do
    steps/decode_sgmm2_rescore.sh --cmd "$decode_cmd" --iter $iter \
      --transform-dir exp/tri4b/decode_3g_${test} data/lang_test_3g data/${test} exp/sgmm2_5a/decode_3g_${test} \
      exp/sgmm2_5a_mmi_b0.1_m0.9/decode_3g_${test}_it$iter &
  done




# The next commands are the same thing on all the si284 data.

# SGMM system on the si284 data [sgmm5b]
  steps/train_ubm.sh --cmd "$train_cmd" \
    600 data/train data/lang exp/tri4b_ali exp/ubm5b || exit 1;

  steps/train_sgmm2.sh --cmd "$train_cmd" \
   11000 25000 data/train data/lang exp/tri4b_ali \
    exp/ubm5b/final.ubm exp/sgmm2_5b || exit 1;

  (
    utils/mkgraph.sh data/lang_test_3g exp/sgmm2_5b exp/sgmm2_5b/graph_3g
    steps/decode_sgmm2.sh --nj 7 --cmd "$decode_cmd" --transform-dir exp/tri4b/decode_3g_${test} \
      exp/sgmm2_5b/graph_3g data/${test} exp/sgmm2_5b/decode_3g_${test}
#    steps/decode_sgmm2.sh --nj 8 --cmd "$decode_cmd" --transform-dir exp/tri4b/decode_tgpr_eval92 \
#      exp/sgmm2_5b/graph_tgpr data/test_eval92 exp/sgmm2_5b/decode_tgpr_eval92

    utils/mkgraph.sh data/lang_test_4g exp/sgmm2_5b exp/sgmm2_5b/graph_4g || exit 1;
    steps/decode_sgmm2.sh --nj 7 --cmd "$decode_cmd" --transform-dir exp/tri4b/decode_4g_${test} \
      exp/sgmm2_5b/graph_4g data/${test} exp/sgmm2_5b/decode_4g_${test}
#    steps/decode_sgmm2.sh --nj 8 --cmd "$decode_cmd" --transform-dir exp/tri4b/decode_bd_tgpr_eval92 \
#      exp/sgmm2_5b/graph_bd_tgpr data/test_eval92 exp/sgmm2_5b/decode_bd_tgpr_eval92
  ) &


 # This shows how you would build and test a quinphone SGMM2 system, but
  (
   steps/train_sgmm2.sh --cmd "$train_cmd" \
      --context-opts "--context-width=5 --central-position=2" \
    11000 25000 data/train data/lang exp/tri4b_ali \
     exp/ubm5b/final.ubm exp/sgmm2_5c || exit 1;
   # Decode from lattices in exp/sgmm2_5b
    steps/decode_sgmm2_fromlats.sh --cmd "$decode_cmd"  --transform-dir exp/tri4b/decode_3g_${test} \
       data/${test} data/lang_test_3g exp/sgmm2_5b/decode_3g_${test} exp/sgmm2_5c/decode_3g_${test} &
    steps/decode_sgmm2_fromlats.sh --cmd "$decode_cmd"  --transform-dir exp/tri4b/decode_4g_${test} \
       data/${test} data/lang_test_4g exp/sgmm2_5b/decode_4g_${test} exp/sgmm2_5c/decode_4g_${test}
  ) &

wait

  steps/align_sgmm2.sh --nj 30 --cmd "$train_cmd" --transform-dir exp/tri4b_ali \
    --use-graphs true --use-gselect true data/train data/lang exp/sgmm2_5b exp/sgmm2_5b_ali

  steps/make_denlats_sgmm2.sh --nj 30 --sub-split 2 --cmd "$decode_cmd" --transform-dir exp/tri4b_ali \
    data/train data/lang exp/sgmm2_5b_ali exp/sgmm2_5b_denlats

  wait

  steps/train_mmi_sgmm2.sh --cmd "$decode_cmd" --transform-dir exp/tri4b_ali --boost 0.1 \
    data/train data/lang exp/sgmm2_5b_ali exp/sgmm2_5b_denlats exp/sgmm2_5b_mmi_b0.1

  for iter in 1 2 3 4; do
    for test in ${test}; do # dev93
      steps/decode_sgmm2_rescore.sh --cmd "$decode_cmd" --iter $iter \
        --transform-dir exp/tri4b/decode_4g_${test} data/lang_test_4g data/${test} exp/sgmm2_5b/decode_4g_${test} \
        exp/sgmm2_5b_mmi_b0.1/decode_4g_${test}_it$iter &
     done
  done

  wait

  steps/train_mmi_sgmm2.sh --cmd "$decode_cmd" --transform-dir exp/tri4b_ali --boost 0.1 \
    --drop-frames true data/train data/lang exp/sgmm2_5b_ali exp/sgmm2_5b_denlats exp/sgmm2_5b_mmi_b0.1_z

  for iter in 1 2 3 4; do
    for test in test ${test}; do
      steps/decode_sgmm2_rescore.sh --cmd "$decode_cmd" --iter $iter \
        --transform-dir exp/tri4b/decode_4g_${test} data/lang_test_4g data/${test} exp/sgmm2_5b/decode_4g_${test} \
        exp/sgmm2_5b_mmi_b0.1_z/decode_4g_${test}_it$iter &
     done
  done

wait

# Examples of combining some of the best decodings: SGMM+MMI with
# MMI+fMMI on a conventional system.

local/score_combine.sh data/${test} \
   data/lang_test_4g \
   exp/tri4b_fmmi_a/decode_3g_${test}_it8 \
   exp/sgmm2_5b_mmi_b0.1/decode_4g_${test}_it3 \
   exp/combine_tri4b_fmmi_a_sgmm2_5b_mmi_b0.1/decode_4g_${test}_it8_3
