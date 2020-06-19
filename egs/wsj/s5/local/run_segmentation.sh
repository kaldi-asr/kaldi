#!/usr/bin/env bash

# Copyright 2014  Guoguo Chen
# Apache 2.0

# This script demonstrates how to re-segment long audios into short segments.
# The basic idea is to decode with an existing in-domain acoustic model, and a
# bigram language model built from the reference, and then work out the
# segmentation from a ctm like file.
# See the script local/run_segmentation_long_utts.sh for 
# a more sophesticated approach using Smith-Waterman alignment
# to align decoded hypothesis and parts of imperfect long-transcripts # retrieved using TF-IDF document similarities.

stage=0

. utils/parse_options.sh

. ./cmd.sh
. ./path.sh

if [ $stage -le 0 ]; then
  local/append_utterances.sh data/train_si284 data/train_si284_long
  steps/cleanup/split_long_utterance.sh \
    --seg-length 30 --overlap-length 5 \
    data/train_si284_long data/train_si284_split
fi

if [ $stage -le 1 ]; then
  steps/make_mfcc.sh --cmd "$train_cmd" --nj 64 \
                     data/train_si284_split exp/make_mfcc/train_si284_split mfcc || exit 1;
  steps/compute_cmvn_stats.sh data/train_si284_split \
    exp/make_mfcc/train_si284_split mfcc || exit 1;
fi

if [ $stage -le 2 ]; then
  steps/cleanup/make_segmentation_graph.sh \
    --cmd "$mkgraph_cmd" --nj 32 \
    data/train_si284_split/ data/lang_nosp exp/tri2b/ \
    exp/tri2b/graph_train_si284_split || exit 1;
fi

if [ $stage -le 3 ]; then
  steps/cleanup/decode_segmentation.sh \
    --nj 64 --cmd "$decode_cmd" --skip-scoring true \
    exp/tri2b/graph_train_si284_split \
    data/train_si284_split exp/tri2b/decode_train_si284_split || exit 1;
fi

if [ $stage -le 4 ]; then
  steps/get_ctm.sh --cmd "$decode_cmd" data/train_si284_split \
    exp/tri2b/graph_train_si284_split exp/tri2b/decode_train_si284_split
fi

if [ $stage -le 5 ]; then
  steps/cleanup/make_segmentation_data_dir.sh --wer-cutoff 0.9 \
    --min-sil-length 0.5 --max-seg-length 15 --min-seg-length 1 \
    exp/tri2b/decode_train_si284_split/score_10/train_si284_split.ctm \
   data/train_si284_split data/train_si284_reseg_a
fi

# Now, use the re-segmented data for training.
if [ $stage -le 6 ]; then
  steps/make_mfcc.sh --cmd "$train_cmd" --nj 64 \
    data/train_si284_reseg_a exp/make_mfcc/train_si284_reseg_a mfcc || exit 1;
  steps/compute_cmvn_stats.sh data/train_si284_reseg_a \
                              exp/make_mfcc/train_si284_reseg_a mfcc || exit 1;
fi

if [ $stage -le 7 ]; then
  steps/align_fmllr.sh --nj 20 --cmd "$train_cmd" \
    data/train_si284_reseg_a data/lang_nosp exp/tri3b exp/tri3b_ali_si284_reseg_a || exit 1;
fi

if [ $stage -le 8 ]; then
  steps/train_sat.sh  --cmd "$train_cmd" \
    4200 40000 data/train_si284_reseg_a \
    data/lang_nosp exp/tri3b_ali_si284_reseg_a exp/tri4c || exit 1;
fi


if [ $stage -le 9 ]; then
  utils/mkgraph.sh data/lang_nosp_test_tgpr exp/tri4c exp/tri4c/graph_tgpr || exit 1;
  steps/decode_fmllr.sh --nj 10 --cmd "$decode_cmd" \
    exp/tri4c/graph_tgpr data/test_dev93 exp/tri4c/decode_tgpr_dev93 || exit 1;
  steps/decode_fmllr.sh --nj 8 --cmd "$decode_cmd" \
    exp/tri4c/graph_tgpr data/test_eval92 exp/tri4c/decode_tgpr_eval92 || exit 1;
fi
