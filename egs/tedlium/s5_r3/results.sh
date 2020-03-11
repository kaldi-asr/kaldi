#!/usr/bin/env bash

# The output of this script (after successfully running ./run.sh) can be found in the RESULTS file.

filter_regexp=.
[ $# -ge 1 ] && filter_regexp=$1

# kaldi scoring,
for x in exp/{mono,tri,sgmm,nnet,dnn,lstm}*/decode*; do
  [ -d $x ] && grep WER $x/wer_* | utils/best_wer.sh;
done 2>/dev/null
for x in exp/chain*/*/decode*; do
  [ -d $x ] && grep WER $x/wer_* | utils/best_wer.sh;
done 2>/dev/null | grep $filter_regexp

# sclite scoring,
for x in exp/{mono,tri,sgmm,nnet,dnn,lstm}*/decode*; do
  [ -d $x ] && grep Sum $x/score_*/*.sys | utils/best_wer.sh;
done 2>/dev/null | grep $filter_regexp
for x in exp/chain*/*/decode*; do
  [ -d $x ] && grep Sum $x/score_*/*.sys | utils/best_wer.sh;
done 2>/dev/null | grep $filter_regexp

exit 0

