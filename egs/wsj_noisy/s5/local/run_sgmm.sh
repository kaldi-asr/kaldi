#!/bin/bash

# This script is invoked from ../run.sh
# It contains some SGMM-related scripts that I am breaking out of the main run.sh for clarity.

. cmd.sh

# SGMM system on si84 data [sgmm5a].  Note: the system we aligned from used the si284 data for
# training, but this shouldn't have much effect.

(
  steps/align_fmllr.sh --nj 30 --cmd "$train_cmd" \
    data/train_si84 data/lang exp/tri4b exp/tri4b_ali_si84 || exit 1;

  steps/train_ubm.sh --cmd "$train_cmd" \
    400 data/train_si84 data/lang exp/tri4b_ali_si84 exp/ubm5a || exit 1;

  steps/train_sgmm.sh --cmd "$train_cmd" \
    3500 10000 data/train_si84 data/lang exp/tri4b_ali_si84 \
    exp/ubm5a/final.ubm exp/sgmm5a || exit 1;

  (
    utils/mkgraph.sh data/lang_test_tgpr exp/sgmm5a exp/sgmm5a/graph_tgpr
    steps/decode_sgmm.sh --nj 10 --cmd "$decode_cmd" --transform-dir exp/tri4b/decode_tgpr_dev93 \
      exp/sgmm5a/graph_tgpr data/test_dev93 exp/sgmm5a/decode_tgpr_dev93
  ) &

  steps/align_sgmm.sh --nj 30 --cmd "$train_cmd" --transform-dir exp/tri4b_ali_si84 \
    --use-graphs true --use-gselect true data/train_si84 data/lang exp/sgmm5a exp/sgmm5a_ali_si84 || exit 1;
  steps/make_denlats_sgmm.sh --nj 30 --sub-split 30 --cmd "$decode_cmd" --transform-dir exp/tri4b_ali_si84 \
    data/train_si84 data/lang exp/sgmm5a_ali_si84 exp/sgmm5a_denlats_si84

  steps/train_mmi_sgmm.sh --cmd "$decode_cmd" --transform-dir exp/tri4b_ali_si84 --boost 0.1 \
    data/train_si84 data/lang exp/sgmm5a_ali_si84 exp/sgmm5a_denlats_si84 exp/sgmm5a_mmi_b0.1

  for iter in 1 2 3 4; do
    steps/decode_sgmm_rescore.sh --cmd "$decode_cmd" --iter $iter \
      --transform-dir exp/tri4b/decode_tgpr_dev93 data/lang_test_tgpr data/test_dev93 exp/sgmm5a/decode_tgpr_dev93 \
      exp/sgmm5a_mmi_b0.1/decode_tgpr_dev93_it$iter &
  done

  steps/train_mmi_sgmm.sh --cmd "$decode_cmd" --transform-dir exp/tri4b_ali_si84 --boost 0.1 \
   --update-opts "--cov-min-value=0.9" data/train_si84 data/lang exp/sgmm5a_ali_si84 exp/sgmm5a_denlats_si84 exp/sgmm5a_mmi_b0.1_m0.9

  for iter in 1 2 3 4; do
    steps/decode_sgmm_rescore.sh --cmd "$decode_cmd" --iter $iter \
      --transform-dir exp/tri4b/decode_tgpr_dev93 data/lang_test_tgpr data/test_dev93 exp/sgmm5a/decode_tgpr_dev93 \
      exp/sgmm5a_mmi_b0.1_m0.9/decode_tgpr_dev93_it$iter &
  done

) &


(
# The next commands are the same thing on all the si284 data.

# SGMM system on the si284 data [sgmm5b]
  steps/train_ubm.sh --cmd "$train_cmd" \
    600 data/train_si284 data/lang exp/tri4b_ali_si284 exp/ubm5b || exit 1;

  steps/train_sgmm.sh --cmd "$train_cmd" \
    5500 25000 data/train_si284 data/lang exp/tri4b_ali_si284 \
    exp/ubm5b/final.ubm exp/sgmm5b || exit 1;

  (
    utils/mkgraph.sh data/lang_test_tgpr exp/sgmm5b exp/sgmm5b/graph_tgpr
    steps/decode_sgmm.sh --nj 10 --cmd "$decode_cmd" --transform-dir exp/tri4b/decode_tgpr_dev93 \
      exp/sgmm5b/graph_tgpr data/test_dev93 exp/sgmm5b/decode_tgpr_dev93
    steps/decode_sgmm.sh --nj 8 --cmd "$decode_cmd" --transform-dir exp/tri4b/decode_tgpr_eval92 \
      exp/sgmm5b/graph_tgpr data/test_eval92 exp/sgmm5b/decode_tgpr_eval92

    utils/mkgraph.sh data/lang_test_bd_tgpr exp/sgmm5b exp/sgmm5b/graph_bd_tgpr || exit 1;
    steps/decode_sgmm.sh --nj 10 --cmd "$decode_cmd" --transform-dir exp/tri4b/decode_bd_tgpr_dev93 \
      exp/sgmm5b/graph_bd_tgpr data/test_dev93 exp/sgmm5b/decode_bd_tgpr_dev93
    steps/decode_sgmm.sh --nj 8 --cmd "$decode_cmd" --transform-dir exp/tri4b/decode_bd_tgpr_eval92 \
      exp/sgmm5b/graph_bd_tgpr data/test_eval92 exp/sgmm5b/decode_bd_tgpr_eval92
  ) &

  steps/align_sgmm.sh --nj 30 --cmd "$train_cmd" --transform-dir exp/tri4b_ali_si284 \
    --use-graphs true --use-gselect true data/train_si284 data/lang exp/sgmm5b exp/sgmm5b_ali_si284 

  steps/make_denlats_sgmm.sh --nj 30 --sub-split 30 --cmd "$decode_cmd" --transform-dir exp/tri4b_ali_si284 \
    data/train_si284 data/lang exp/sgmm5b_ali_si284 exp/sgmm5b_denlats_si284

  steps/train_mmi_sgmm.sh --cmd "$decode_cmd" --transform-dir exp/tri4b_ali_si284 --boost 0.1 \
    data/train_si284 data/lang exp/sgmm5b_ali_si284 exp/sgmm5b_denlats_si284 exp/sgmm5b_mmi_b0.1

  for iter in 1 2 3 4; do
    for test in dev93 eval92; do
      steps/decode_sgmm_rescore.sh --cmd "$decode_cmd" --iter $iter \
        --transform-dir exp/tri4b/decode_tgpr_${test} data/lang_test_tgpr data/test_${test} exp/sgmm5b/decode_tgpr_${test} \
        exp/sgmm5b_mmi_b0.1/decode_tgpr_${test}_it$iter &

      steps/decode_sgmm_rescore.sh --cmd "$decode_cmd" --iter $iter \
        --transform-dir exp/tri4b/decode_bd_tgpr_${test} data/lang_test_bd_tgpr data/test_${test} exp/sgmm5b/decode_bd_tgpr_${test} \
        exp/sgmm5b_mmi_b0.1/decode_bd_tgpr_${test}_it$iter &
     done
  done
) &



# Train quinphone SGMM system. 

steps/train_sgmm.sh  --cmd "$train_cmd" \
   --context-opts "--context-width=5 --central-position=2" \
   5500 25000 data/train_si284 data/lang exp/tri4b_ali_si284 \
   exp/ubm5b/final.ubm exp/sgmm5c || exit 1;

# Decode from lattices in exp/sgmm5a/decode_tgpr_dev93.
steps/decode_sgmm_fromlats.sh --cmd "$decode_cmd"  --transform-dir exp/tri4b/decode_tgpr_dev93 \
   data/test_dev93 data/lang_test_tgpr exp/sgmm5a/decode_tgpr_dev93 exp/sgmm5c/decode_tgpr_dev93 
