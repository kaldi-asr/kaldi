#!/bin/bash

. ./path.sh

# Example script for lookahead composition

lm=tgmed
am=exp/chain_cleaned/tdnn_1d_sp
testset=test_clean

# %WER 4.86 [ 2553 / 52576, 315 ins, 222 del, 2016 sub ] exp/chain_cleaned/tdnn_1d_sp/decode_test_clean_lookahead/wer_11_0.0
# %WER 4.79 [ 2518 / 52576, 279 ins, 292 del, 1947 sub ] exp/chain_cleaned/tdnn_1d_sp/decode_test_clean_lookahead_arpa/wer_11_0.0
# %WER 4.82 [ 2532 / 52576, 286 ins, 290 del, 1956 sub ] exp/chain_cleaned/tdnn_1d_sp/decode_test_clean_lookahead_arpa_fast/wer_11_0.0
# %WER 4.86 [ 2553 / 52576, 314 ins, 222 del, 2017 sub ] exp/chain_cleaned/tdnn_1d_sp/decode_test_clean_lookahead_base/wer_11_0.0
# %WER 4.86 [ 2553 / 52576, 315 ins, 222 del, 2016 sub ] exp/chain_cleaned/tdnn_1d_sp/decode_test_clean_lookahead_static/wer_11_0.0


# Speed
#
# base       0.18 xRT
# static     0.18 xRT
# lookahead  0.29 xRT
# arpa       0.35 xRT
# arpa_fast  0.21 xRT

# Graph size
#
# Base                 476 Mb
# Static               621 Mb
# Lookahead            48 Mb HCL + 77 Mb Grammar
# Lookahead + OpenGrm  48 Mb HCL + 42 Mb Grammar

if [ ! -f "${KALDI_ROOT}/tools/openfst/lib/libfstlookahead.so" ]; then
    echo "Missing ${KALDI_ROOT}/tools/openfst/lib/libfstlookahead.so"
    echo "Make sure you compiled openfst with lookahead support. Run make in ${KALDI_ROOT}/tools after git pull."
    exit 1
fi
if [ ! -f "${KALDI_ROOT}/tools/openfst/bin/ngramread" ]; then
    echo "You appear to not have OpenGRM tools installed. Missing ${KALDI_ROOT}/tools/openfst/bin/ngramread"
    echo "cd to $KALDI_ROOT/tools and run extras/install_opengrm.sh."
    exit 1
fi
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

# Compile arpa graph
utils/mkgraph_lookahead.sh --self-loop-scale 1.0 --compose-graph \
    data/lang_test_${lm}_base ${am} data/local/lm/lm_tgmed.arpa.gz ${am}/graph_${lm}_lookahead_arpa

# Decode with runtime composition
steps/nnet3/decode_lookahead.sh --nj 20 \
    --acwt 1.0 --post-decode-acwt 10.0 \
    --online-ivector-dir exp/nnet3_cleaned/ivectors_${testset}_hires \
    ${am}/graph_${lm}_lookahead_arpa data/${testset}_hires ${am}/decode_${testset}_lookahead_arpa

# Decode with runtime composition and tuned beams
steps/nnet3/decode_lookahead.sh --nj 20 \
    --beam 12.0 --max-active 3000 \
    --acwt 1.0 --post-decode-acwt 10.0 \
    --online-ivector-dir exp/nnet3_cleaned/ivectors_${testset}_hires \
    ${am}/graph_${lm}_lookahead_arpa data/${testset}_hires ${am}/decode_${testset}_lookahead_arpa_fast
