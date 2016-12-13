#!/bin/bash

# Copyright 2016  Vimal Manohar
# Apache 2.0

set -e -o pipefail

# This script demonstrates how to re-segment long audios into short segments.
# The basic idea is to decode with an existing in-domain acoustic model, and a
# bigram language model built from the reference, and then work out the
# segmentation from a ctm like file.

## %WER results. 

## Baseline with manual transcripts
# %WER 7.87 [ 444 / 5643, 114 ins, 25 del, 305 sub ] exp/tri4a/decode_nosp_tgpr_eval92/wer_13_1.0
# %WER 11.84 [ 975 / 8234, 187 ins, 107 del, 681 sub ] exp/tri4a/decode_nosp_tgpr_dev93/wer_17_0.5

## Baseline using local/run_segmentation.sh
# %WER 7.76 [ 438 / 5643, 119 ins, 22 del, 297 sub ] exp/tri4c/decode_tgpr_eval92/wer_14_0.5
# %WER 12.41 [ 1022 / 8234, 216 ins, 96 del, 710 sub ] exp/tri4c/decode_tgpr_dev93/wer_17_0.0

## Training directly on segmented data directory train_si284_reseg
# %WER 7.55 [ 426 / 5643, 97 ins, 25 del, 304 sub ] exp/tri4c_reseg/decode_nosp_tgpr_eval92/wer_16_1.0
# %WER 7.87 [ 444 / 5643, 113 ins, 25 del, 306 sub ] exp/tri3c_reseg/decode_nosp_tgpr_eval92/wer_14_1.0

# %WER 11.94 [ 983 / 8234, 170 ins, 123 del, 690 sub ] exp/tri4c_reseg/decode_nosp_tgpr_dev93/wer_17_1.0
# %WER 12.19 [ 1004 / 8234, 208 ins, 112 del, 684 sub ] exp/tri3c_reseg/decode_nosp_tgpr_dev93/wer_17_0.5

## Using additional stage of cleanup.
## Cleanup does not help here. The transcripts are already pretty good.
# %WER 7.71 [ 435 / 5643, 110 ins, 26 del, 299 sub ] exp/tri4d_cleaned_a/decode_nosp_tgpr_eval92/wer_13_1.0
# %WER 7.46 [ 421 / 5643, 110 ins, 18 del, 293 sub ] exp/tri4d_cleaned_c/decode_nosp_tgpr_eval92/wer_15_0.5
# %WER 7.90 [ 446 / 5643, 106 ins, 32 del, 308 sub ] exp/tri4d_cleaned_b/decode_nosp_tgpr_eval92/wer_16_1.0

# %WER 12.16 [ 1001 / 8234, 208 ins, 105 del, 688 sub ] exp/tri4d_cleaned_a/decode_nosp_tgpr_dev93/wer_15_0.5
# %WER 12.24 [ 1008 / 8234, 213 ins, 100 del, 695 sub ] exp/tri4d_cleaned_c/decode_nosp_tgpr_dev93/wer_15_0.5
# %WER 12.35 [ 1017 / 8234, 219 ins, 112 del, 686 sub ] exp/tri4d_cleaned_b/decode_nosp_tgpr_dev93/wer_17_0.5

. ./cmd.sh
. ./path.sh

###############################################################################
## Simulate unsegmented data directory.
###############################################################################
local/append_utterances.sh data/train_si284 data/train_si284_long

steps/make_mfcc.sh --cmd "$train_cmd" --nj 32 \
  data/train_si284_long exp/make_mfcc/train_si284_long mfcc || exit 1
steps/compute_cmvn_stats.sh data/train_si284_long \
  exp/make_mfcc/train_si284_long mfcc

###############################################################################
# Segment long recordings using TF-IDF retrieval of reference text 
# for uniformly segmented audio chunks based on Smith-Waterman alignment.
# Use a model trained on train_si84 (tri2b)
###############################################################################
steps/cleanup/segment_long_utterances.sh --cmd "$train_cmd" \
  --max-segment-duration 30 --overlap-duration 5 \
  --num-neighbors-to-search 0 --nj 80 \
  exp/tri2b data/lang_nosp data/train_si284_long data/train_si284_reseg \
  exp/segment_long_utts_train_si284

steps/compute_cmvn_stats.sh data/train_si284_reseg \
  exp/make_mfcc/train_si284_reseg mfcc
utils/fix_data_dir.sh data/train_si284_reseg

###############################################################################
# Train new model on segmented data directory starting from the same model
# used for segmentation. (tri2b)
###############################################################################

# Align tri2b system with reseg data
steps/align_si.sh  --nj 40 --cmd "$train_cmd" \
  data/train_si284_reseg \
  data/lang_nosp exp/tri2b exp/tri2b_ali_si284_reseg  || exit 1;

# Train SAT system on reseg data
steps/train_sat.sh --cmd "$train_cmd" 4200 40000 \
  data/train_si284_reseg data/lang_nosp \
  exp/tri2b_ali_si284_reseg exp/tri3c_reseg

(
utils/mkgraph.sh data/lang_nosp_test_tgpr \
  exp/tri3c_reseg exp/tri3c_reseg/graph_nosp_tgpr || exit 1;
steps/decode_fmllr.sh --nj 10 --cmd "$decode_cmd" \
  exp/tri3c_reseg/graph_nosp_tgpr data/test_dev93 \
  exp/tri3c_reseg/decode_nosp_tgpr_dev93 || exit 1;
steps/decode_fmllr.sh --nj 8 --cmd "$decode_cmd" \
  exp/tri3c_reseg/graph_nosp_tgpr data/test_eval92 \
  exp/tri3c_reseg/decode_nosp_tgpr_eval92 || exit 1;
) &

###############################################################################
# Train new model on segmented data directory starting from a better model
# (tri3b)
###############################################################################

# Align tri3b system with reseg data
steps/align_fmllr.sh --nj 40 --cmd "$train_cmd" \
  data/train_si284_reseg data/lang_nosp exp/tri3b \
  exp/tri3b_ali_si284_reseg

# Train SAT system on reseg data
steps/train_sat.sh --cmd "$train_cmd" 4200 40000 \
  data/train_si284_reseg data/lang_nosp \
  exp/tri3b_ali_si284_reseg exp/tri4c_reseg

(
 utils/mkgraph.sh data/lang_nosp_test_tgpr \
   exp/tri4c_reseg exp/tri4c_reseg/graph_nosp_tgpr || exit 1;
 steps/decode_fmllr.sh --nj 10 --cmd "$decode_cmd" \
   exp/tri4c_reseg/graph_nosp_tgpr data/test_dev93 \
   exp/tri4c_reseg/decode_nosp_tgpr_dev93 || exit 1;
 steps/decode_fmllr.sh --nj 8 --cmd "$decode_cmd" \
   exp/tri4c_reseg/graph_nosp_tgpr data/test_eval92 \
   exp/tri4c_reseg/decode_nosp_tgpr_eval92 || exit 1;
) &

###############################################################################
# cleaned_a : Cleanup the segmented data directory using tri3b model.
###############################################################################

steps/cleanup/clean_and_segment_data.sh --cmd "$train_cmd" \
  --nj 80 \
  data/train_si284_reseg data/lang_nosp \
  exp/tri3b_ali_si284_reseg exp/tri3b_work_si284_reseg data/train_si284_reseg_cleaned_a

###############################################################################
# Train new model on the cleaned_a data directory
###############################################################################

# Align tri3b system with cleaned data
steps/align_fmllr.sh --nj 40 --cmd "$train_cmd" \
  data/train_si284_reseg_cleaned_a data/lang_nosp exp/tri3b \
  exp/tri3b_ali_si284_reseg_cleaned_a

# Train SAT system on cleaned data
steps/train_sat.sh --cmd "$train_cmd" 4200 40000 \
  data/train_si284_reseg_cleaned_a data/lang_nosp \
  exp/tri3b_ali_si284_reseg_cleaned_a exp/tri4d_cleaned_a

(
 utils/mkgraph.sh data/lang_nosp_test_tgpr \
   exp/tri4d_cleaned_a exp/tri4d_cleaned_a/graph_nosp_tgpr || exit 1;
 steps/decode_fmllr.sh --nj 10 --cmd "$decode_cmd" \
   exp/tri4d_cleaned_a/graph_nosp_tgpr data/test_dev93 \
   exp/tri4d_cleaned_a/decode_nosp_tgpr_dev93 || exit 1;
 steps/decode_fmllr.sh --nj 8 --cmd "$decode_cmd" \
   exp/tri4d_cleaned_a/graph_nosp_tgpr data/test_eval92 \
   exp/tri4d_cleaned_a/decode_nosp_tgpr_eval92 || exit 1;
) &

###############################################################################
# cleaned_b : Cleanup the segmented data directory using the tri3c_reseg
# model, which is a like a first-pass model trained on the resegmented data.
###############################################################################

steps/cleanup/clean_and_segment_data.sh --cmd "$train_cmd" \
  --nj 80 \
  data/train_si284_reseg data/lang_nosp \
  exp/tri3c_reseg exp/tri3c_reseg_work_si284_reseg \
  data/train_si284_reseg_cleaned_b

###############################################################################
# Train new model on the cleaned_b data directory
###############################################################################

# Align tri3c_reseg system with cleaned data
steps/align_fmllr.sh --nj 40 --cmd "$train_cmd" \
  data/train_si284_reseg_cleaned_b data/lang_nosp exp/tri3c_reseg \
  exp/tri3c_reseg_ali_si284_reseg_cleaned_b

# Train SAT system on cleaned data
steps/train_sat.sh --cmd "$train_cmd" 4200 40000 \
  data/train_si284_reseg_cleaned_b data/lang_nosp \
  exp/tri3c_reseg_ali_si284_reseg_cleaned_b exp/tri4d_cleaned_b

(
 utils/mkgraph.sh data/lang_nosp_test_tgpr \
   exp/tri4d_cleaned_b exp/tri4d_cleaned_b/graph_nosp_tgpr || exit 1;
 steps/decode_fmllr.sh --nj 10 --cmd "$decode_cmd" \
   exp/tri4d_cleaned_b/graph_nosp_tgpr data/test_dev93 \
   exp/tri4d_cleaned_b/decode_nosp_tgpr_dev93 || exit 1;
 steps/decode_fmllr.sh --nj 8 --cmd "$decode_cmd" \
   exp/tri4d_cleaned_b/graph_nosp_tgpr data/test_eval92 \
   exp/tri4d_cleaned_b/decode_nosp_tgpr_eval92 || exit 1;
) &

###############################################################################
# cleaned_c : Cleanup the segmented data directory using the tri4c_reseg
# model, which is a like a first-pass model trained on the resegmented data.
###############################################################################

steps/cleanup/clean_and_segment_data.sh --cmd "$train_cmd" \
  --nj 80 \
  data/train_si284_reseg data/lang_nosp \
  exp/tri4c_reseg exp/tri4c_reseg_work_si284_reseg \
  data/train_si284_reseg_cleaned_c

###############################################################################
# Train new model on the cleaned_c data directory
###############################################################################

# Align tri4c_reseg system with cleaned data
steps/align_fmllr.sh --nj 40 --cmd "$train_cmd" \
  data/train_si284_reseg_cleaned_c data/lang_nosp exp/tri4c_reseg \
  exp/tri4c_reseg_ali_si284_reseg_cleaned_c

# Train SAT system on cleaned data
steps/train_sat.sh --cmd "$train_cmd" 4200 40000 \
  data/train_si284_reseg_cleaned_c data/lang_nosp \
  exp/tri4c_reseg_ali_si284_reseg_cleaned_c exp/tri4d_cleaned_c

(
 utils/mkgraph.sh data/lang_nosp_test_tgpr \
   exp/tri4d_cleaned_c exp/tri4d_cleaned_c/graph_nosp_tgpr || exit 1;
 steps/decode_fmllr.sh --nj 10 --cmd "$decode_cmd" \
   exp/tri4d_cleaned_c/graph_nosp_tgpr data/test_dev93 \
   exp/tri4d_cleaned_c/decode_nosp_tgpr_dev93 || exit 1;
 steps/decode_fmllr.sh --nj 8 --cmd "$decode_cmd" \
   exp/tri4d_cleaned_c/graph_nosp_tgpr data/test_eval92 \
   exp/tri4d_cleaned_c/decode_nosp_tgpr_eval92 || exit 1;
) &
