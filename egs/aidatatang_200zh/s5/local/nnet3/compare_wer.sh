#!/usr/bin/env bash
# Copyright 2018  Emotech LTD (Author: Xuechen Liu)

# compare wer between diff. models in aidatatang_200zh nnet3 directory
# exemplar usage: local/nnet3/compare_wer.sh exp/nnet3/tdnn_sp
# note: this script is made quite general since we kinda wanna give more flexibility to
#       users on adding affix for their own use when training models.

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
    echo "compare_wer.sh: internal error"
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
  wer=$([ -d $x ] && grep WER $x/decode_test/cer_* | utils/best_wer.sh | awk '{print $2}')
  printf "% 10s" $wer
done
echo

# so how about online WER?
if $include_online; then
  echo -n "# WER(%)[online]       "
  for x in $*; do
    set_names $x
    wer=$(cat ${x}_online/decode_test/cer_* | utils/best_wer.sh | awk '{print $2}')
    printf "% 10s" $wer
  done
  echo
  echo -n "# WER(%)[per-utt]      "
  for x in $*; do
    set_names $x
    wer_per_utt=$(cat ${x}_online/decode_test_per_utt/cer_* | utils/best_wer.sh | awk '{print $2}')
    printf "% 10s" $wer_per_utt
  done
  echo
fi

# print log for train & validation
echo -n "# Final train prob     "
for x in $*; do
  prob=$(grep Overall $x/log/compute_prob_train.combined.log | grep log-like | awk '{printf($8)}' | cut -c1-7)
  printf "% 10s" $prob
done
echo

echo -n "# Final valid prob     "
for x in $*; do
  prob=$(grep Overall $x/log/compute_prob_valid.combined.log | grep log-like | awk '{printf($8)}' | cut -c1-7)
  printf "% 10s" $prob
done
echo
