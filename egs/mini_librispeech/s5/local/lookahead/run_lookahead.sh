#!/bin/bash

. ./path.sh

# Example script for lookahead composition

lm=tgmed
am=exp/chain_online_cmn/tdnn1k_sp
testset=dev_clean_2

# %WER 10.32 [ 2078 / 20138, 201 ins, 275 del, 1602 sub ] exp/chain_online_cmn/tdnn1k_sp/decode_dev_clean_2_lookahead_base/wer_10_0.5
# %WER 10.29 [ 2073 / 20138, 200 ins, 272 del, 1601 sub ] exp/chain_online_cmn/tdnn1k_sp/decode_dev_clean_2_lookahead_static/wer_10_0.5
# %WER 10.25 [ 2064 / 20138, 192 ins, 277 del, 1595 sub ] exp/chain_online_cmn/tdnn1k_sp/decode_dev_clean_2_lookahead/wer_10_0.5
# %WER 10.24 [ 2063 / 20138, 187 ins, 290 del, 1586 sub ] exp/chain_online_cmn/tdnn1k_sp/decode_dev_clean_2_lookahead_arpa/wer_10_0.5
# %WER 10.29 [ 2072 / 20138, 228 ins, 242 del, 1602 sub ] exp/chain_online_cmn/tdnn1k_sp/decode_dev_clean_2_lookahead_arpa_fast/wer_9_0.5

# Speed
#
# base       0.29 xRT
# static     0.31 xRT
# lookahead  0.77 xRT
# arpa       1.03 xRT
# arpa_fast  0.31 xRT

# Graph size
#
# Base                 461 Mb
# Static               587 Mb
# Lookahead            44 Mb HCL + 77 Mb Grammar
# Lookahead + OpenGrm  44 Mb HCL + 42 Mb Grammar

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
    --online-ivector-dir exp/nnet3_online_cmn/ivectors_${testset}_hires \
    ${am}/graph_${lm}_lookahead_base data/${testset}_hires ${am}/decode_${testset}_lookahead_base

utils/mkgraph_lookahead.sh --self-loop-scale 1.0 --remove-oov --compose-graph \
    data/lang_test_${lm}_base ${am} ${am}/graph_${lm}_lookahead

# Decode with statically composed lookahead graph
steps/nnet3/decode.sh --nj 20 \
    --acwt 1.0 --post-decode-acwt 10.0 \
    --online-ivector-dir exp/nnet3_online_cmn/ivectors_${testset}_hires \
    ${am}/graph_${lm}_lookahead data/${testset}_hires ${am}/decode_${testset}_lookahead_static

# Decode with runtime composition
steps/nnet3/decode_lookahead.sh --nj 20 \
    --acwt 1.0 --post-decode-acwt 10.0 \
    --online-ivector-dir exp/nnet3_online_cmn/ivectors_${testset}_hires \
    ${am}/graph_${lm}_lookahead data/${testset}_hires ${am}/decode_${testset}_lookahead

# Compile arpa graph
utils/mkgraph_lookahead.sh --self-loop-scale 1.0 --compose-graph \
    data/lang_test_${lm}_base ${am} data/local/lm/lm_tgmed.arpa.gz ${am}/graph_${lm}_lookahead_arpa

# Decode with runtime composition
steps/nnet3/decode_lookahead.sh --nj 20 \
    --acwt 1.0 --post-decode-acwt 10.0 \
    --online-ivector-dir exp/nnet3_online_cmn/ivectors_${testset}_hires \
    ${am}/graph_${lm}_lookahead_arpa data/${testset}_hires ${am}/decode_${testset}_lookahead_arpa

# Decode with runtime composition and tuned beams
steps/nnet3/decode_lookahead.sh --nj 20 \
    --beam 12.0 --max-active 3000 \
    --acwt 1.0 --post-decode-acwt 10.0 \
    --online-ivector-dir exp/nnet3_online_cmn/ivectors_${testset}_hires \
    ${am}/graph_${lm}_lookahead_arpa data/${testset}_hires ${am}/decode_${testset}_lookahead_arpa_fast
