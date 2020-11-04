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
nnet3_affix=_cleaned # affix for the chain directory name
affix=1d   # affix for the TDNN directory name
rnnlm_rescore=false

# End configuration section
. ./utils/parse_options.sh

. ./cmd.sh
. ./path.sh

# RNNLM rescore options
ngram_order=4 # approximate the lattice-rescoring by limiting the max-ngram-order
              # if it's set, it merges histories in the lattice if they share
              # the same ngram history and this prevents the lattice from 
              # exploding exponentially
pruned_rescore=true
rnnlm_dir=exp/rnnlm_lstm_1a

dir=exp/chain${nnet3_affix}/tdnn_${affix}

# Get dev and eval set names from the test_sets
dev_set=$( echo $test_sets | cut -d " " -f1 )
eval_set=$( echo $test_sets | cut -d " " -f2 )


set -e # exit on error

##########################################################################
# DECODING: we perform 2 stage decoding.
##########################################################################

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
        exp/chain${nnet3_affix}/tdnn_${affix}
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
      --dev exp/chain${nnet3_affix}/tdnn_${affix}/decode_${dev_set}_oracle_2stage \
      --eval exp/chain${nnet3_affix}/tdnn_${affix}/decode_${eval_set}_oracle_2stage
fi

############################################################################
# RNNLM rescoring
############################################################################
if $rnnlm_rescore; then
  if [ $stage -le 2 ]; then
    echo "$0: Perform RNNLM lattice-rescoring"
    pruned=
    ac_model_dir=exp/chain${nnet3_affix}/tdnn_${affix}
    if $pruned_rescore; then
      pruned=_pruned
    fi
    for decode_set in $test_sets; do
      decode_dir=${ac_model_dir}/decode_${decode_set}_oracle_2stage
      # Lattice rescoring
      rnnlm/lmrescore$pruned.sh \
          --cmd "$decode_cmd --mem 8G" \
          --weight 0.45 --max-ngram-order $ngram_order \
          $lang_dir $rnnlm_dir \
          data/${decode_set}_oracle_hires ${decode_dir} \
          ${ac_model_dir}/decode_${decode_set}_oracle_2stage_rescore
    done
  fi
  if [ $stage -le 3 ]; then
    echo "$0: WERs after rescoring with $rnnlm_dir"
    local/score_reco_oracle.sh \
        --dev exp/chain${nnet3_affix}/tdnn_${affix}/decode_${dev_set}_oracle_2stage${rescore_dir_suffix} \
        --eval exp/chain${nnet3_affix}/tdnn_${affix}/decode_${eval_set}_oracle_2stage${rescore_dir_suffix}
  fi
fi
