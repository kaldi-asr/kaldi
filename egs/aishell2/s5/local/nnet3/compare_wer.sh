#!/usr/bin/env bash

# Copyright 2018 Emotech LTD (Author: Xuechen LIU)
# Apache 2.0

# compare wer between diff. models in aishell2 nnet3 directory

set -e
. ./cmd.sh
. ./path.sh

if [ $# == 0 ]; then
  echo "Usage: $0: <dir1> [<dir2> ... ]"
  echo "e.g.: $0 exp/nnet3/tdnn_sp exp/nnet3/tdnn_sp_pr"
  exit 1
fi

echo "# $0 $*"

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
  wer=$([ -d $x ] && grep WER $x/decode_test/cer_* | utils/best_wer.sh | awk '{print $2}')
  printf "% 10s" $wer
done
echo

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
