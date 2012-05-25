#!/bin/bash

# This script is invoked from ../run.sh
# It contains some SGMM-related scripts that I am breaking out of the main run.sh for clarity.


# SGMM system on si84 data [sgmm5a].  Note: the system we aligned from used the si284 data for
# training, but this shouldn't have much effect.

steps/align_fmllr.sh --nj 30 --cmd "$train_cmd" \
  data/train_si84 data/lang exp/tri4b exp/tri4b_ali_si84 || exit 1;

steps/train_ubm.sh --cmd "$train_cmd" \
  400 data/train_si84 data/lang exp/tri4b_ali_si84 exp/ubm5a || exit 1;

steps/train_sgmm.sh --cmd "$train_cmd" \
  3500 10000 data/train_si84 data/lang exp/tri4b_ali_si84 \
   exp/ubm5b/final.ubm exp/sgmm5a || exit 1;

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
    --transform-dir exp/tri4b/decode_tgpr_eval92 data/lang data/test exp/sgmm5a/decode exp/sgmm4a_mmi_b0.2/decode_it$iter &
 done




# The next commands are the same thing on all the si284 data.

# SGMM system on the si284 data [sgmm5b]
steps/train_ubm.sh --cmd "$train_cmd" \
  600 data/train_si284 data/lang exp/tri4b_ali_si284 exp/ubm5b || exit 1;

steps/train_sgmm.sh --cmd "$train_cmd" --phn-dim 50 \
  5500 25000 data/train_si284 data/lang exp/tri4b_ali_si284 \
  exp/ubm5b/final.ubm exp/sgmm5b || exit 1;

 (
  utils/mkgraph.sh data/lang_test_tgpr exp/sgmm5b exp/sgmm5b/graph_tgpr
  steps/decode_sgmm.sh --nj 10 --cmd "$decode_cmd" --transform-dir exp/tri4b/decode_tgpr_dev93 \
    exp/sgmm5b/graph_tgpr data/test_dev93 exp/sgmm5b/decode_tgpr_dev93
 ) &

 steps/align_sgmm.sh --nj 30 --cmd "$train_cmd" --transform-dir exp/tri4b_ali_si284 \
  --use-graphs true --use-gselect true data/train_si284 data/lang exp/sgmm5a exp/sgmm5a_ali_si284 
