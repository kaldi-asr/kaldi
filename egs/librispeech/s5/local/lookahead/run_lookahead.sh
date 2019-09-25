#!/bin/bash

. path.sh

# Example script for lookahead composition

lm=tgmed
am=exp/chain_cleaned/tdnn_1d_sp
testset=test_clean

# %WER 4.86 [ 2553 / 52576, 314 ins, 222 del, 2017 sub ] exp/chain_cleaned/tdnn_1d_sp/decode_test_clean_lookahead_base/wer_11_0.0
# %WER 4.96 [ 2608 / 52576, 342 ins, 218 del, 2048 sub ] exp/chain_cleaned/tdnn_1d_sp/decode_test_clean_lookahead_static/wer_11_0.0
# %WER 4.96 [ 2608 / 52576, 342 ins, 218 del, 2048 sub ] exp/chain_cleaned/tdnn_1d_sp/decode_test_clean_lookahead/wer_11_0.0
# %WER 4.91 [ 2583 / 52576, 310 ins, 267 del, 2006 sub ] exp/chain_cleaned/tdnn_1d_sp/decode_test_clean_lookahead_arpa/wer_11_0.0

# Speed
#
# base       0.186742 xRT
# static     0.164752 xRT
# lookahead  0.289464 xRT
# arpa       0.272674 xRT

# Graph size
#
# Base                 476 Mb
# Static               675 Mb
# Lookahead            22 Mb HCL + 77 Mb Grammar
# Lookahead + OpenGrm  22 Mb HCL + 42 Mb Grammar

export LD_LIBRARY_PATH=${KALDI_ROOT}/tools/openfst/lib/fst

# Baseline
utils/format_lm.sh data/lang data/local/lm/lm_${lm}.arpa.gz \
    data/local/dict/lexicon.txt data/lang_test_${lm}_base

utils/mkgraph.sh --self-loop-scale 1.0 --remove-oov \
    data/lang_test_${lm}_base ${am} ${am}/graph_${lm}_lookahead_base

steps/nnet3/decode.sh --nj 20 \
    --acwt 1.0 --post-decode-acwt 10.0 \
    --online-ivector-dir exp/nnet3_cleaned/ivectors_${testset}_hires \
    ${am}/graph_${lm}_lookahead_base data/${testset}_hires ${am}/decode_${testset}_lookahead_base

utils/mkgraph_lookahead.sh --self-loop-scale 1.0 --remove-oov --compose-graph \
    data/lang_test_${lm}_base ${am} ${am}/graph_${lm}_lookahead

# Decode with statically composed lookahead graph
steps/nnet3/decode.sh --nj 20 \
    --acwt 1.0 --post-decode-acwt 10.0 \
    --online-ivector-dir exp/nnet3_cleaned/ivectors_${testset}_hires \
    ${am}/graph_${lm}_lookahead data/${testset}_hires ${am}/decode_${testset}_lookahead_static

# Decode with runtime composition
steps/nnet3/decode_lookahead.sh --nj 20 \
    --acwt 1.0 --post-decode-acwt 10.0 \
    --online-ivector-dir exp/nnet3_cleaned/ivectors_${testset}_hires \
    ${am}/graph_${lm}_lookahead data/${testset}_hires ${am}/decode_${testset}_lookahead

utils/mkgraph_lookahead_arpa.sh --self-loop-scale 1.0 --compose-graph \
    data/lang_test_${lm}_base data/local/lm/lm_tgmed.arpa.gz ${am} ${am}/graph_${lm}_lookahead_arpa

# Decode with runtime composition
steps/nnet3/decode_lookahead.sh --nj 20 \
    --acwt 1.0 --post-decode-acwt 10.0 \
    --online-ivector-dir exp/nnet3_cleaned/ivectors_${testset}_hires \
    ${am}/graph_${lm}_lookahead_arpa data/${testset}_hires ${am}/decode_${testset}_lookahead_arpa
