#!/bin/bash

# Copyright 2014  Guoguo Chen
# Apache 2.0

set -e -o pipefail

# This script demonstrates how to re-segment long audios into short segments.
# The basic idea is to decode with an existing in-domain acoustic model, and a
# bigram language model built from the reference, and then work out the
# segmentation from a ctm like file.

. ./cmd.sh
. ./path.sh

if false; then
  local/append_utterances.sh data/train_si284 data/train_si284_long

steps/make_mfcc.sh --cmd "$train_cmd" --nj 32 \
  data/train_si284_long exp/make_mfcc/train_si284_long mfcc || exit 1
steps/compute_cmvn_stats.sh data/train_si284_long \
  exp/make_mfcc/train_si284_long mfcc

  # Use a model trained on train_si84
  steps/cleanup/segment_long_utterances.sh --cmd "$train_cmd" \
    --max-segment-duration 30 --overlap-duration 5 \
    --num-neighbors-to-search 0 --nj 80 \
    exp/tri2b data/lang_nosp data/train_si284_long data/train_si284_reseg \
    exp/segment_long_utts_train_si284

  steps/compute_cmvn_stats.sh data/train_si284_reseg \
    exp/make_mfcc/train_si284_reseg mfcc
  utils/fix_data_dir.sh data/train_si284_reseg

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
fi

steps/cleanup/clean_and_segment_data.sh --cmd "$train_cmd" \
  --nj 80 \
  data/train_si284_reseg data/lang_nosp \
  exp/tri3b_ali_si284_reseg exp/tri3b_work_si284_reseg data/train_si284_reseg_cleaned_a

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

steps/cleanup/clean_and_segment_data.sh --cmd "$train_cmd" \
  --nj 80 \
  data/train_si284_reseg data/lang_nosp \
  exp/tri3c_reseg exp/tri3c_reseg_work_si284_reseg \
  data/train_si284_reseg_cleaned_b

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

steps/cleanup/clean_and_segment_data.sh --cmd "$train_cmd" \
  --nj 80 \
  data/train_si284_reseg data/lang_nosp \
  exp/tri4c_reseg exp/tri4c_reseg_work_si284_reseg \
  data/train_si284_reseg_cleaned_c

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
