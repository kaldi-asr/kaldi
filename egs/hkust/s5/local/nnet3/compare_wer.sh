#!/bin/bash
# Copyright 2018  Emotech LTD (Author: Xuechen Liu)

# compare wer between diff. models in hkust nnet3 directory
# exemplar usage: local/nnet3/compare_wer_general.sh exp/nnet3/tdnn_sp exp/nnet3/tdnn_sp_pr43
# note: this script is made quite general since we kinda wanna give more flexibility to
#       users on adding affix for their own use when training models. This script applies
#       to both nnet3 and chain models trained within hkust corpus (it may not be applicable
#       for other egs such as mini-librispeech, where some modifications are needed for grepping).

set -e
. ./cmd.sh
. ./path.sh

if [ $# == 0 ]; then
  echo "Usage: $0: [--online] <dir1> [<dir2> ... ]"
  echo "e.g.: $0 exp/nnet3/tdnn_sp exp/nnet3/tdnn_sp_pr"
  exit 1
fi

echo "# $0 $*"

include_online=false
if [ "$1" == "--online" ]; then
  include_online=true
  shift
fi

set_names() {
  if [ $# != 1 ]; then
    echo "compare_wer_general.sh: internal error"
    exit 1  # exit the program
  fi
  dirname=$(echo $1 | cut -d: -f1)
}

# print model names
echo -n "# Model               "
for x in $*; do
  printf "% 10s" " $(basename $x)"
done
echo

# print decode WER results
echo -n "# WER(%)               "
for x in $*; do
  set_names $x
  wer=$([ -d $x ] && grep WER $x/decode/cer_* | utils/best_wer.sh | awk '{print $2}')
  printf "% 10s" $wer
done
echo

# so how about online WER?
if $include_online; then
  echo -n "# WER(%)[online]       "
  for x in $*; do
    set_names $x
    wer=$(cat ${x}_online/decode/cer_* | utils/best_wer.sh | awk '{print $2}')
    printf "% 10s" $wer
  done
  echo
  echo -n "# WER(%)[per-utt]      "
  for x in $*; do
    set_names $x
    wer_per_utt=$(cat ${x}_online/decode_per_utt/cer_* | utils/best_wer.sh | awk '{print $2}')
    printf "% 10s" $wer_per_utt
  done
  echo
fi

# print log for train & validation. note: we know for hkust nnet&chain model the final prob in log
# directory is marked with letter 'n' (like, 'combined' and 'final') so we did below. It's quite a 
# hacky but convenient way
echo -n "# Final train prob     "
for x in $*; do
  prob=$(grep Overall $x/log/compute_prob_train.combined.log | grep -v xent | awk '{printf($8)}' | cut -c1-7)
  printf "% 10s" $prob
done
echo

echo -n "# Final valid prob     "
for x in $*; do
  prob=$(grep Overall $x/log/compute_prob_valid.combined.log | grep -v xent | awk '{printf($8)}' | cut -c1-7)
  printf "% 10s" $prob
done
echo
