#!/bin/bash

# Copyright 2013  Bagher BabaAli

. ./cmd.sh 
[ -f path.sh ] && . ./path.sh

echo ============================================================================
echo "                Data & Lexicon & Language Preparation                     "
echo ============================================================================

timit=/export/corpora5/LDC/LDC93S1/timit/TIMIT

local/timit_data_prep.sh $timit  || exit 1;

local/timit_prepare_dict.sh || exit 1;


utils/prepare_lang.sh --position-dependent-phones false --num-sil-states 3 \
 data/local/dict "sil" data/local/lang_tmp data/lang || exit 1;

local/timit_format_data.sh || exit 1;

echo ============================================================================
echo "        MFCC Feature Extration & CMVN for Training and Test set           "
echo ============================================================================

# Now make MFCC features.
mfccdir=mfcc
for x in test train; do 
 steps/make_mfcc.sh --cmd "$train_cmd" --nj 30 \
  data/$x exp/make_mfcc/$x $mfccdir || exit 1;
 steps/compute_cmvn_stats.sh data/$x exp/make_mfcc/$x $mfccdir || exit 1;
done

vecho ============================================================================
echo "                     MonoPhone Training & Decoding                        "
echo ============================================================================

steps/train_mono.sh  --nj 30 --cmd "$train_cmd" data/train data/lang exp/mono || exit 1;

utils/mkgraph.sh --mono data/lang_test_bg exp/mono exp/mono/graph_bg || exit 1;

steps/decode.sh --nj 30  --cmd "$decode_cmd" \
 exp/mono/graph_bg data/test exp/mono/decode_bg_test || exit 1;

echo ============================================================================
echo "           tri1 : Deltas + Delta-Deltas Training & Decoding               "
echo ============================================================================

steps/align_si.sh --boost-silence 1.25 --nj 30 --cmd "$train_cmd" \
 data/train data/lang exp/mono exp/mono_ali || exit 1;

# Train tri1, which is deltas + delta-deltas, on train data.
steps/train_deltas.sh --cmd "$train_cmd" \
 2500 15000 data/train data/lang exp/mono_ali exp/tri1 || exit 1;

utils/mkgraph.sh data/lang_test_bg exp/tri1 exp/tri1/graph_bg || exit 1;

steps/decode.sh --nj 30 --cmd "$decode_cmd" \
 exp/tri1/graph_bg data/test exp/tri1/decode_bg_test || exit 1;

echo ============================================================================
echo "                 tri2 : LDA + MLLT Training & Decoding                    "
echo ============================================================================

steps/align_si.sh --nj 30 --cmd "$train_cmd" \
  data/train data/lang exp/tri1 exp/tri1_ali_train || exit 1;

steps/train_lda_mllt.sh --cmd "$train_cmd" \
 --splice-opts "--left-context=3 --right-context=3" \
 2500 15000 data/train data/lang exp/tri1_ali_train exp/tri2 || exit 1;

utils/mkgraph.sh data/lang_test_bg exp/tri2 exp/tri2/graph_bg || exit 1;

steps/decode.sh --nj 30 --cmd "$decode_cmd" \
 exp/tri2/graph_bg data/test exp/tri2/decode_bg_test || exit 1;

echo ============================================================================
echo "              tri3 : LDA + MLLT + SAT Training & Decoding                 "
echo ============================================================================

# Align tri2 system with train data.
steps/align_si.sh  --nj 30 --cmd "$train_cmd" \
 --use-graphs true data/train data/lang exp/tri2 exp/tri2_ali_train  || exit 1;

# From tri2 system, train tri3 which is LDA + MLLT + SAT.
steps/train_sat.sh --cmd "$train_cmd" \
 2500 15000 data/train data/lang exp/tri2_ali_train exp/tri3 || exit 1;

utils/mkgraph.sh data/lang_test_bg exp/tri3 exp/tri3/graph_bg || exit 1;

steps/decode_fmllr.sh --nj 30 --cmd "$decode_cmd" \
 exp/tri3/graph_bg data/test exp/tri3/decode_bg_test || exit 1;

echo ============================================================================
echo "            fMMI + MMI Training & Decoding on top of tri3                 "
echo ============================================================================

# Train and test MMI, and boosted MMI, on tri3 (LDA+MLLT+SAT on all the data).
# From tri3 system, align all train data.
steps/align_fmllr.sh --nj 30 --cmd "$train_cmd" \
 data/train data/lang exp/tri3 exp/tri3_ali_train || exit 1;

steps/make_denlats.sh --nj 30 --sub-split 30 --cmd "$train_cmd" \
 --transform-dir exp/tri3_ali_train data/train data/lang \
 exp/tri3 exp/tri3_denlats_train || exit 1;

steps/train_mmi.sh --cmd "$train_cmd" --boost 0.1 \
 data/train data/lang exp/tri3_ali_train exp/tri3_denlats_train \
 exp/tri3_mmi_b0.1  || exit 1;

steps/decode.sh --nj 30 --cmd "$decode_cmd" --transform-dir exp/tri3/decode_bg_test \
 exp/tri3/graph_bg data/test exp/tri3_mmi_b0.1/decode_bg_test


#first, train UBM for fMMI experiments.
steps/train_diag_ubm.sh --silence-weight 0.5 --nj 30 --cmd "$train_cmd" \
 600 data/train data/lang exp/tri3_ali_train exp/dubm3

# Next, fMMI+MMI.
steps/train_mmi_fmmi.sh  --boost 0.1 --cmd "$train_cmd" \
 data/train data/lang exp/tri3_ali_train exp/dubm3 exp/tri3_denlats_train \
 exp/tri3_fmmi_a || exit 1;

for iter in 1 2 3 4; do
 steps/decode_fmmi.sh --nj 30  --cmd "$decode_cmd" --iter $iter \
   --transform-dir exp/tri3/decode_bg_test  exp/tri3/graph_bg data/test \
  exp/tri3_fmmi_a/decode_bg_test_it$iter 
done

# fMMI + mmi with indirect differential.
steps/train_mmi_fmmi_indirect.sh --boost 0.1 --cmd "$train_cmd" \
 data/train data/lang exp/tri3_ali_train exp/dubm3 exp/tri3_denlats_train \
 exp/tri3_fmmi_indirect || exit 1;

for iter in 1 2 3 4; do
 steps/decode_fmmi.sh --nj 30  --cmd "$decode_cmd" --iter $iter \
   --transform-dir exp/tri3/decode_bg_test  exp/tri3/graph_bg data/test \
  exp/tri3_fmmi_indirect/decode_bg_test_it$iter 
done

echo ============================================================================
echo "                        SGMM2 Training & Decoding                         "
echo ============================================================================

steps/align_fmllr.sh --nj 30 --cmd "$train_cmd" \
 data/train data/lang exp/tri3 exp/tri3_ali_train || exit 1;

steps/train_ubm.sh --cmd "$train_cmd" \
 400 data/train data/lang exp/tri3_ali_train exp/ubm4 || exit 1;

steps/train_sgmm2.sh --cmd "$train_cmd" 7000 9000 \
 data/train data/lang exp/tri3_ali_train exp/ubm4/final.ubm exp/sgmm2_4 || exit 1;

utils/mkgraph.sh data/lang_test_bg exp/sgmm2_4 exp/sgmm2_4/graph_bg || exit 1;

steps/decode_sgmm2.sh --nj 30 --cmd "$decode_cmd"\
 --transform-dir exp/tri3/decode_bg_test exp/sgmm2_4/graph_bg data/test \
 exp/sgmm2_4/decode_bg_test || exit 1;

echo ============================================================================
echo "                    MMI + SGMM2 Training & Decoding                       "
echo ============================================================================

steps/align_sgmm2.sh --nj 30 --cmd "$train_cmd" \
 --transform-dir exp/tri3_ali_train --use-graphs true --use-gselect true data/train \
 data/lang exp/sgmm2_4 exp/sgmm2_4_ali_train || exit 1;

steps/make_denlats_sgmm2.sh --nj 30 --sub-split 30 --cmd "$decode_cmd"\
 --transform-dir exp/tri3_ali_train  data/train data/lang exp/sgmm2_4_ali_train \
 exp/sgmm2_4_denlats_train || exit 1;

steps/train_mmi_sgmm2.sh --cmd "$decode_cmd" \
 --transform-dir exp/tri3_ali_train --boost 0.1 --zero-if-disjoint true \
 data/train data/lang exp/sgmm2_4_ali_train exp/sgmm2_4_denlats_train \
 exp/sgmm2_4_mmi_b0.1_z || exit 1;

for iter in 1 2 3 4; do
  steps/decode_sgmm2_rescore.sh --cmd "$decode_cmd" --iter $iter \
   --transform-dir exp/tri3/decode_bg_test data/lang_test_bg data/test \
   exp/sgmm2_4/decode_bg_test exp/sgmm2_4_mmi_b0.1_z/decode_bg_test_it$iter || exit 1;
done

echo ============================================================================
echo "                  Combining some of the Best Decodings                    "
echo ============================================================================

local/score_combine.sh data/test data/lang_test_bg \
 exp/tri3_fmmi_a/decode_bg_test_it1 exp/sgmm2_4_mmi_b0.1_z/decode_bg_test_it1 \
 exp/combine_tri3b_fmmi_a_sgmm2_4_mmi_b0.1_z/decode_bg_test_it1_1 || exit 1;

# Checking MBR decode of baseline:
cp -r -T exp/sgmm2_4_mmi_b0.1_z/decode_bg_test_it1{,.mbr}
local/score_mbr.sh data/test data/lang_test_bg \
 exp/sgmm2_4_mmi_b0.1_z/decode_bg_test_it1.mbr || exit 1;

echo ============================================================================
echo "                    Getting Results [see RESULTS file]                    "
echo ============================================================================

for x in exp/*/decode*; do
  [ -d $x ] && grep WER $x/wer_* | utils/best_wer.sh
done

exit 0;




