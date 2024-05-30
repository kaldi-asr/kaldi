#!/usr/bin/env bash

filter_regexp=.
[ $# -ge 1 ] && filter_regexp=$1

for x in exp/*/decode*; do 
  [ -d $x ] && grep WER $x/wer_* | utils/best_wer.sh; 
done 2>/dev/null
  
for x in exp/*{mono,tri,sgmm,nnet,dnn,lstm,chain}*/decode*; do 
  [ -d $x ] && grep Sum $x/score_*/*.sys | utils/best_wer.sh; 
done 2>/dev/null | grep $filter_regexp

for x in exp/*{nnet,dnn,lstm,chain}*/*/decode*; do 
  [ -d $x ] && grep Sum $x/score_*/*.sys | utils/best_wer.sh; 
done 2>/dev/null | grep $filter_regexp

exit 0

