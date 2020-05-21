#!/usr/bin/env bash
#
# Based mostly on the TED-LIUM and Switchboard recipe
#
# Copyright  2017  Johns Hopkins University (Author: Shinji Watanabe and Yenda Trmal)
# Apache 2.0
#
# This script performs recognition with oracle speaker and segment information

# Begin configuration section.
decode_nj=20
stage=0
test_sets=
lang_dir=
lm_suffix=

# End configuration section
. ./utils/parse_options.sh

. ./cmd.sh
. ./path.sh

affix=1d   # affix for the TDNN directory name
dir=exp/chain${nnet3_affix}/tdnn_${affix}_sp


set -e # exit on error

##########################################################################
# DECODING: we perform 2 stage decoding.
##########################################################################

nnet3_affix=_cleaned

if [ $stage -le 0 ]; then
  # First the options that are passed through to run_ivector_common.sh
  # (some of which are also used in this script directly).

  # The rest are configs specific to this script.  Most of the parameters
  # are just hardcoded at this level, in the commands below.
  echo "$0: decode data..."
  
  # training options
  # training chunk-options
  chunk_width=150,110,100
  # we don't need extra left/right context for TDNN systems.
  chunk_left_context=0
  chunk_right_context=0
  
  utils/mkgraph.sh \
      --self-loop-scale 1.0 $lang_dir \
      $dir $dir/graph${lm_suffix} || exit 1;

  frames_per_chunk=$(echo $chunk_width | cut -d, -f1)
  rm $dir/.error 2>/dev/null || true

  for data in $test_sets; do
    (
      local/nnet3/decode.sh --affix 2stage --pass2-decode-opts "--min-active 1000" \
        --acwt 1.0 --post-decode-acwt 10.0 \
        --frames-per-chunk 150 --nj $decode_nj \
        --ivector-dir exp/nnet3${nnet3_affix} \
        data/${data}_oracle $lang_dir \
        $dir/graph${lm_suffix} \
        exp/chain${nnet3_affix}/tdnn_${affix}_sp
    ) || touch $dir/.error &
  done
  wait
  [ -f $dir/.error ] && echo "$0: there was a problem while decoding" && exit 1
fi

##########################################################################
# Scoring: here we obtain wer per condition and overall WER
##########################################################################

if [ $stage -le 1 ]; then
  # please specify both dev and eval set directories so that the search parameters
  # (insertion penalty and language model weight) will be tuned using the dev set
  local/score_reco_oracle.sh \
      --dev exp/chain${nnet3_affix}/tdnn_${affix}_sp/decode_dev_oracle_2stage \
      --eval exp/chain${nnet3_affix}/tdnn_${affix}_sp/decode_eval_oracle_2stage
fi
