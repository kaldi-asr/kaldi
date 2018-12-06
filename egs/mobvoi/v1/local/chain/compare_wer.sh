#!/bin/bash

# this script is used for comparing decoding results between systems.
# e.g. local/chain/compare_wer.sh exp/chain/cnn{1a,1b}

# Copyright      2017  Chun Chieh Chang
#                2017  Ashish Arora

if [ $# == 0 ]; then
  echo "Usage: $0: <dir1> [<dir2> ... ]"
  echo "e.g.: $0 exp/chain/cnn{1a,1b}"
  exit 1
fi
. ./path.sh

echo "# $0 $*"
used_epochs=false

echo -n "# System                      "
for x in $*; do   printf "% 10s" " $(basename $x)";   done
echo

echo -n "# Precision/Recall(%) dev     "
for x in $*; do
  precision=$(cat $x/decode_dev/scoring_kaldi/best_metrics | awk '{print $2*100}')
  recall=$(cat $x/decode_dev/scoring_kaldi/best_metrics | awk '{print $4*100}')
  printf "%6s/%3s" $precision $recall
done
echo

echo -n "# Precision/Recall(%) eval    "
for x in $*; do
  precision=$(cat $x/decode_eval/scoring_kaldi/best_metrics | awk '{print $2*100}')
  recall=$(cat $x/decode_eval/scoring_kaldi/best_metrics | awk '{print $4*100}')
  printf "%6s/%3s" $precision $recall
done
echo


echo -n "# WER dev                     "
for x in $*; do
  wer=$(cat $x/decode_dev/scoring_kaldi/best_wer | awk '{print $2}')
  printf "% 10s" $wer
done
echo

echo -n "# WER eval                    "
for x in $*; do
  wer=$(cat $x/decode_eval/scoring_kaldi/best_wer | awk '{print $2}')
  printf "% 10s" $wer
done
echo

if $used_epochs; then
  exit 0;  # the diagnostics aren't comparable between regular and discriminatively trained systems.
fi

echo -n "# Final train prob           "
for x in $*; do
  prob=$(grep Overall $x/log/compute_prob_train.final.log | grep -v xent | awk '{printf("%.4f", $8)}')
  printf "% 10s" $prob
done
echo

echo -n "# Final valid prob           "
for x in $*; do
  prob=$(grep Overall $x/log/compute_prob_valid.final.log | grep -v xent | awk '{printf("%.4f", $8)}')
  printf "% 10s" $prob
done
echo

echo -n "# Final train prob (xent)    "
for x in $*; do
  prob=$(grep Overall $x/log/compute_prob_train.final.log | grep -w xent | awk '{printf("%.4f", $8)}')
  printf "% 10s" $prob
done
echo

echo -n "# Final valid prob (xent)    "
for x in $*; do
  prob=$(grep Overall $x/log/compute_prob_valid.final.log | grep -w xent | awk '{printf("%.4f", $8)}')
  printf "% 10s" $prob
done
echo

echo -n "# Parameters                 "
for x in $*; do
  params=$(nnet3-info $x/final.mdl 2>/dev/null | grep num-parameters | cut -d' ' -f2 | awk '{printf "%0.2fM\n",$1/1000000}')
  printf "% 10s" $params
done
echo
