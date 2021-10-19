#!/usr/bin/env bash
# Copyright 2021  ASLP, NWPU (Author: Hang Lyu)
#                 Mobvoi Inc (Author: Binbin Zhang)
# Apache 2.0

# To accommodate with the setups of other toolkits, we give up the techniques
# about SpecAug and Ivector in this script.
# 1c use multi-stream cnn model.

# Set -e here so that we catch if any executable fails immediately
set -euo pipefail

stage=15
train_nj=20
decode_nj=20
train_set="train_l"
nnet3_affix=_cleaned

# The rest are configs specific to this script.  Most of the parameters
# are just hardcoded at this level, in the commands below.
affix=_1c   # affix for the TDNN directory name
decode_iter=

# decode options
test_sets="test_aishell1"
test_online_decoding=false  # if true, it will run the last decoding stage.


# End configuration section.
echo "$0 $@"  # Print the command line for logging

. ./cmd.sh
. ./path.sh
. ./utils/parse_options.sh

dir=exp/${train_set}/chain${nnet3_affix}/cnn_tdnn${affix}_sp



if [ $stage -le 15 ]; then
  steps/make_mfcc.sh --nj $decode_nj --mfcc-config conf/mfcc_hires.conf \
    --cmd "$decode_cmd" data/test_aishell1_hires || exit 1;
  steps/compute_cmvn_stats.sh data/test_aishell1_hires || exit 1;
  utils/fix_data_dir.sh data/test_aishell1_hires
fi

if [ $stage -le 16 ]; then
  steps/online/nnet2/extract_ivectors_online.sh --cmd "$decode_cmd" \
    --nj $decode_nj \
    data/test_aishell1_hires exp/${train_set}/nnet3${nnet3_affix}/extractor \
    exp/${train_set}/nnet3${nnet3_affix}/ivectors_test_aishell1_hires || exit 1;
fi

if [ $stage -le 17 ]; then
  rm $dir/.error 2>/dev/null || true
  for part_set in $test_sets; do
      (
      steps/nnet3/decode.sh --acwt 1.0 --post-decode-acwt 10.0 \
          --nj $decode_nj --cmd "$decode_cmd" \
          --online-ivector-dir exp/${train_set}/nnet3${nnet3_affix}/ivectors_${part_set}_hires \
          $dir/graph data/${part_set}_hires $dir/decode_${part_set}${decode_iter:+_$decode_iter} || exit 1
      ) || touch $dir/.error
  done
  wait
  if [ -f $dir/.error ]; then
    echo "$0: something went wrong in decoding"
    exit 1
  fi
fi

if [ $stage -le 19 ]; then
  # decode with rnnlm
  # If an rnnlm has been provided, we should set the "stage" to 4 for testing.
  ./local/wenetspeech_run_rnnlm.sh --stage 4 \
    --train-stage -10 \
    --ngram-order 5 \
    --num-epoch 8 \
    --num-jobs-initial 1 \
    --num-jobs-final 8 \
    --words-per-split 400000 \
    --text data/corpus/lm_text \
    --ac-model-dir $dir \
    --test-sets "$test_sets" \
    --decode-iter "$decode_iter" \
    --lang data/lang_test \
    --dir exp/rnnlm
fi

exit 0;
