#!/bin/bash

# Copyright 2018  Emotech LTD (Author: Xuechen LIU)
# Apache 2.0

# compare wer between diff. models in aishell2 chain directory

set -e
. ./cmd.sh
. ./path.sh

if [ $# == 0 ]; then
  echo "Usage: $0: <dir1> [<dir2> ... ]"
  echo "e.g.: $0 exp/chain/tdnn_7h_sp"
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

# print number of parameters
echo -n "# Num. of params        "
for x in $*; do
  set_names $x
  params=$(steps/info/chain_dir_info.pl "$x" | grep -o 'num-params=[0-9]*\.[0-9]*M' | cut -d'=' -f2-)
  printf "% 10s\n" $params
done

# print decode WER results
echo -n "# WER(%)               "
for x in $*; do
  set_names $x
  wer=$([ -d $x ] && grep WER $x/decode_test/cer_* | utils/best_wer.sh | awk '{print $2}')
  printf "% 10s" $wer
done
echo

# print final log prob for train & validation
echo -n "# Final train prob     "
for x in $*; do
  prob=$(grep Overall $x/log/compute_prob_train.final.log | grep -v xent | awk '{printf($8)}' | cut -c1-7)
  printf "% 10s" $prob
done
echo

echo -n "# Final valid prob     "
for x in $*; do
  prob=$(grep Overall $x/log/compute_prob_valid.final.log | grep -v xent | awk '{printf($8)}' | cut -c1-7)
  printf "% 10s" $prob
done
echo

# do the same for xent objective
echo -n "# Final train prob (xent)"
for x in $*; do
  prob=$(grep Overall $x/log/compute_prob_train.final.log | grep -w xent | awk '{printf("%.4f", $8)}')
  printf "% 10s" $prob
done
echo

echo -n "# Final valid prob (xent)"
for x in $*; do
  prob=$(grep Overall $x/log/compute_prob_valid.final.log | grep -w xent | awk '{printf("%.4f", $8)}')
  printf "% 10s" $prob
done
echo
