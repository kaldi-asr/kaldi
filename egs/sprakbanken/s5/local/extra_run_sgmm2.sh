#!/bin/bash

# This script is invoked from ../run.sh
# It contains some SGMM-related scripts that I am breaking out of the main run.sh for clarity.

. cmd.sh

# Note: you might want to try to give the option --spk-dep-weights=false to train_sgmm2.sh;
# this takes out the "symmetric SGMM" part which is not always helpful.

# SGMM system on si84 data [sgmm5a].  Note: the system we aligned from used the si284 data for
# training, but this shouldn't have much effect.

# LM suffix
uid="" #$1

# Test set id
test=$1 # $2


  steps/align_fmllr.sh --nj 50 --cmd "$train_cmd" \
    data/train data/lang_$uid exp/tri4b_$uid exp/tri4b_${uid}_ali || exit 1;

  steps/train_ubm.sh --cmd "$train_cmd" \
    600 data/train data/lang_$uid exp/tri4b_${uid}_ali exp/ubm5a_$uid || exit 1;

  steps/train_sgmm2.sh --cmd "$train_cmd" \
    11000 25000 data/train data/lang_$uid exp/tri4b_${uid}_ali \
    exp/ubm5a_$uid/final.ubm exp/sgmm2_5a_$uid || exit 1;

  (
    utils/mkgraph.sh data/lang_test_$uid exp/sgmm2_5a_$uid exp/sgmm2_5a_${uid}/graph_$uid
    steps/decode_sgmm2.sh --nj 7 --cmd "$decode_cmd" --transform-dir exp/tri4b_${uid}/decode_${uid}_$test \
      exp/sgmm2_5a_$uid/graph_${uid} data/$test exp/sgmm2_5a_${uid}/decode_${uid}_$test
  )

  steps/align_sgmm2.sh --nj 50 --cmd "$train_cmd" --transform-dir exp/tri4b_${uid}_ali \
    --use-graphs true --use-gselect true data/train data/lang_${uid} exp/sgmm2_5a_${uid} exp/sgmm2_5a_${uid}_ali || exit 1;
  steps/make_denlats_sgmm2.sh --nj 50 --sub-split 24 --cmd "$decode_cmd" --transform-dir exp/tri4b_${uid}_ali \
    data/train data/lang_${uid} exp/sgmm2_5a_${uid}_ali exp/sgmm2_5a_${uid}_denlats

  steps/train_mmi_sgmm2.sh --cmd "$decode_cmd" --transform-dir exp/tri4b_${uid}_ali --boost 0.1 \
    data/train data/lang_${uid} exp/sgmm2_5a_${uid}_ali exp/sgmm2_5a_${uid}_denlats exp/sgmm2_5a_${uid}_mmi_b0.1

  for iter in 1 2 3 4; do
    steps/decode_sgmm2_rescore.sh --cmd "$decode_cmd" --iter $iter \
      --transform-dir exp/tri4b_${uid}/decode_${uid}_$test data/lang_test_${uid} data/$test exp/sgmm2_5a_${uid}/decode_${uid}_$test \
      exp/sgmm2_5a_${uid}_mmi_b0.1/decode_${uid}_${test}_it$iter 
  done

  steps/train_mmi_sgmm2.sh --cmd "$decode_cmd" --transform-dir exp/tri4b_${uid}_ali --boost 0.1 \
   --update-opts "--cov-min-value=0.9" data/train data/lang_${uid} exp/sgmm2_5a_${uid}_ali exp/sgmm2_5a_${uid}_denlats exp/sgmm2_5a_${uid}_mmi_b0.1_m0.9

  for iter in 1 2 3 4; do
    steps/decode_sgmm2_rescore.sh --cmd "$decode_cmd" --iter $iter \
      --transform-dir exp/tri4b_${uid}/decode_${uid}_$test data/lang_test_${uid} data/$test exp/sgmm2_5a_${uid}/decode_${uid}_$test \
      exp/sgmm2_5a_${uid}_mmi_b0.1_m0.9/decode_${uid}_${test}_it$iter 
  done


exit


(
# The next commands are the same thing on all the si284 data.

# SGMM system on the si284 data [sgmm5b]
  steps/train_ubm.sh --cmd "$train_cmd" \
    600 data/train data/lang_${uid} exp/tri4b_${uid}_ali exp/ubm5b_${uid} || exit 1;

  steps/train_sgmm2.sh --cmd "$train_cmd" \
   11000 25000 data/train data/lang_${uid} exp/tri4b_${uid}_ali \
    exp/ubm5b_${uid}/final.ubm exp/sgmm2_5b_${uid} || exit 1;

  (
    utils/mkgraph.sh data/lang_test_${uid} exp/sgmm2_5b_${uid} exp/sgmm2_5b_${uid}/graph_${uid}
    steps/decode_sgmm2.sh --nj 5 --cmd "$decode_cmd" --transform-dir exp/tri4b_${uid}/decode_${uid}_$test \
      exp/sgmm2_5b_${uid}/graph_${uid} data/$test exp/sgmm2_5b_${uid}/decode_${uid}_$test
    steps/decode_sgmm2.sh --nj 12 --cmd "$decode_cmd" --transform-dir exp/tri4b_${uid}/decode_arpa_dev \
      exp/sgmm2_5b_${uid}/graph_arpa data/dev exp/sgmm2_5b_${uid}/decode_arpa_dev

#    utils/mkgraph.sh data/lang_test_bd_tgpr exp/sgmm2_5b exp/sgmm2_5b/graph_bd_tgpr || exit 1;
#    steps/decode_sgmm2.sh --nj 10 --cmd "$decode_cmd" --transform-dir exp/tri4b/decode_bd_tgpr_dev93 \
#      exp/sgmm2_5b/graph_bd_tgpr data/test_dev93 exp/sgmm2_5b/decode_bd_tgpr_dev93
#    steps/decode_sgmm2.sh --nj 8 --cmd "$decode_cmd" --transform-dir exp/tri4b/decode_bd_tgpr_eval92 \
#      exp/sgmm2_5b/graph_bd_tgpr data/test_eval92 exp/sgmm2_5b/decode_bd_tgpr_eval92
  ) 


 # This shows how you would build and test a quinphone SGMM2 system, but
  (
   steps/train_sgmm2.sh --cmd "$train_cmd" \
      --context-opts "--context-width=5 --central-position=2" \
    11000 25000 data/train data/lang_${uid} exp/tri4b_${uid}_ali \
     exp/ubm5b_${uid}/final.ubm exp/sgmm2_5c_${uid} || exit 1;
   # Decode from lattices in exp/sgmm2_5b
    steps/decode_sgmm2_fromlats.sh --cmd "$decode_cmd"  --transform-dir exp/tri4b_${uid}/decode_${uid}_$test \
       data/$test data/lang_test_${uid} exp/sgmm2_5b_${uid}/decode_${uid}_$test exp/sgmm2_5c_${uid}/decode_${uid}_$test 
#    steps/decode_sgmm2_fromlats.sh --cmd "$decode_cmd"  --transform-dir exp/tri4b/decode_tgpr_eval92 \
#       data/test_eval92 data/lang_test_tgpr exp/sgmm2_5b/decode_tgpr_eval92 exp/sgmm2_5c/decode_tgpr_eval92 
  ) 


  steps/align_sgmm2.sh --nj 24 --cmd "$train_cmd" --transform-dir exp/tri4b_${uid}_ali \
    --use-graphs true --use-gselect true data/train data/lang_${uid} exp/sgmm2_5b_${uid} exp/sgmm2_5b_${uid}_ali 

  steps/make_denlats_sgmm2.sh --nj 24 --sub-split 24 --cmd "$decode_cmd" --transform-dir exp/tri4b_${uid}_ali \
    data/train data/lang_${uid} exp/sgmm2_5b_${uid}_ali exp/sgmm2_5b_${uid}_denlats

  steps/train_mmi_sgmm2.sh --cmd "$decode_cmd" --transform-dir exp/tri4b_${uid}_ali --boost 0.1 \
    data/train data/lang_${uid} exp/sgmm2_5b_${uid}_ali exp/sgmm2_5b_${uid}_denlats exp/sgmm2_5b_${uid}_mmi_b0.1

  for iter in 1 2 3 4; do
      steps/decode_sgmm2_rescore.sh --cmd "$decode_cmd" --iter $iter \
        --transform-dir exp/tri4b_${uid}/decode_${uid}_${test} data/lang_test_${uid} data/${test} exp/sgmm2_5b_${uid}/decode_${uid}_${test} \
        exp/sgmm2_5b_${uid}_mmi_b0.1/decode_${uid}_${test}_it$iter 
  done

  steps/train_mmi_sgmm2.sh --cmd "$decode_cmd" --transform-dir exp/tri4b_${uid}_ali --boost 0.1 \
    --zero-if-disjoint true data/train data/lang_${uid} exp/sgmm2_5b_${uid}_ali exp/sgmm2_5b_${uid}_denlats exp/sgmm2_5b_${uid}_mmi_b0.1_z

  for iter in 1 2 3 4; do
      steps/decode_sgmm2_rescore.sh --cmd "$decode_cmd" --iter $iter \
        --transform-dir exp/tri4b_${uid}/decode_${uid}_${test} data/lang_test_${uid} data/${test} exp/sgmm2_5b_${uid}/decode_${uid}_${test} \
        exp/sgmm2_5b_${uid}_mmi_b0.1_z/decode_${uid}_${test}_it$iter 
  done

) 

wait

# Examples of combining some of the best decodings: SGMM+MMI with
# MMI+fMMI on a conventional system.
 
local/score_combine.sh data/$test \
   data/lang_test_${uid} \
   exp/tri4b_${uid}_fmmi_a/decode_${uid}_${test}_it8 \
   exp/sgmm2_5b_${uid}_mmi_b0.1/decode_${uid}_${test}_it3 \
   exp/combine_tri4b_fmmi_a_sgmm2_5b_mmi_b0.1_${uid}/decode_${uid}_${test}_it8_3


# %WER 4.43 [ 250 / 5643, 41 ins, 12 del, 197 sub ] exp/tri4b_fmmi_a/decode_tgpr_eval92_it8/wer_11
# %WER 3.85 [ 217 / 5643, 35 ins, 11 del, 171 sub ] exp/sgmm2_5b_mmi_b0.1/decode_bd_tgpr_eval92_it3/wer_10
# combined to:
# %WER 3.76 [ 212 / 5643, 32 ins, 12 del, 168 sub ] exp/combine_tri4b_fmmi_a_sgmm2_5b_mmi_b0.1/decode_bd_tgpr_eval92_it8_3/wer_12

exit

# Checking MBR decode of baseline:
cp -r -T exp/sgmm2_5b_mmi_b0.1/decode_bd_tgpr_eval92_it3{,.mbr}
local/score_mbr.sh data/test_eval92 data/lang_test_bd_tgpr exp/sgmm2_5b_mmi_b0.1/decode_bd_tgpr_eval92_it3.mbr
# MBR decoding did not seem to help (baseline was 3.85).  I think this is normal at such low WERs.
%WER 3.86 [ 218 / 5643, 35 ins, 11 del, 172 sub ] exp/sgmm2_5b_mmi_b0.1/decode_bd_tgpr_eval92_it3.mbr/wer_10
