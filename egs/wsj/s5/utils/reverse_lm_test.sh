#!/bin/bash

# Copyright 2012  Brno University of Technology (Author: Mirko Hannemann)
# Apache 2.0

# configuration section
utterances=4
maxlen=30
nbest=10
# end config section

echo "$0 $@"  # Print the command line for logging

[ -f ./path.sh ] && . ./path.sh; # source the path.
. parse_options.sh || exit 1;

if [ $# != 2 ]; then
   echo "Usage: utils/reverse_lm_test.sh [options] <fwd-lm-dir> <bwd-lm-dir>"
   echo "example: utils/reverse_lm_test.sh data/lang_test_tgpr_5k data/lang_test_tgpr_5k.reverse"
   echo "options:"
   echo "  --utterances <int>   number of random test utterances"
   echo "  --maxlen <int>       max number of arcs (words) in utterance"
   echo "  --nbest <int>        compare n best paths"
   exit 1;
fi

test_fwd=$1
test_bwd=$2
nb=`echo $nbest | awk '{print $1-1;}'`

# For each language model the corresponding FST in lang_test_* directory.

echo "compare LM scores using "$test_fwd/G.fst" and "$test_bwd/G.fst

for utt in `seq 1 $utterances`
do
  # generate random sentence with forward language model
  len=1000 # big number
  while [ $len -gt $maxlen ]
  do
    fstrandgen --npath=1 $test_fwd/G.fst | fstprint --acceptor --isymbols=$test_fwd/words.txt --osymbols=$test_fwd/words.txt > sent$utt
    len=`cat sent$utt | wc -l`
  done
  cat sent$utt | awk '(NF>1){if ($3!="#0") {a[length(a)+1]=$3;}} END{printf "utterance:"; for(i=1;i<=length(a);i++) {printf " %s",a[i];} printf "\n";}'  
  
  # get n best paths with forward language model
  cat sent$utt | awk '(NF>1){if ($3!="#0") {a[length(a)+1]=$3;}} END{for(i=1;i<=length(a);i++) {print i-1,i,a[i];} print length(a);}' > sent$utt.forward
  fstcompile --acceptor --isymbols=$test_fwd/words.txt  --osymbols=$test_fwd/words.txt sent$utt.forward > sent$utt.forward.fst
  fstcompose $test_fwd/G.fst sent$utt.forward.fst > sent$utt.composed.forward.fst
  fstshortestpath --nshortest=$nbest sent$utt.composed.forward.fst | fstprint > sent$utt.composed.forward.n

  rm sent$utt.forward.scores 2>/dev/null
  for n in `seq 0 $nb`
  do
    # select path with rank n
    cat sent$utt.composed.forward.n | awk '(NR>'$n' || $1!="0"){print;}' | fstcompile | fstconnect > sent$utt.composed.forward.$n.fst
    fstprint sent$utt.composed.forward.$n.fst > sent$utt.composed.forward.$n
    # compute shortest distance to final states
    fstshortestdistance sent$utt.composed.forward.$n.fst | \
      awk -v list=sent$utt.composed.forward.$n 'BEGIN{mincost=1E5; while (getline < list > 0){if (NF==2) final[$1]=$2; if (NF==1) final[$1]=0.00001;}} \
      { if (final[$1]) { cost=$2+final[$1]; if (cost<mincost) {mincost=cost;} };} END {print mincost;}' \
      >> sent$utt.forward.scores
  done
  
  # get n best paths with reverse language model
  cat sent$utt | awk '(NF>1){if ($3!="#0") {a[length(a)+1]=$3;}} END{for(i=1;i<=length(a);i++) {print i-1,i,a[length(a)-i+1];} print length(a);}' > sent$utt.reverse
  fstcompile --acceptor --isymbols=$test_fwd/words.txt --osymbols=$test_fwd/words.txt sent$utt.reverse > sent$utt.reverse.fst
  fstcompose $test_bwd/G.fst sent$utt.reverse.fst > sent$utt.composed.reverse.fst
  fstshortestpath --nshortest=$nbest sent$utt.composed.reverse.fst | fstprint > sent$utt.composed.reverse.n

  rm sent$utt.reverse.scores 2>/dev/null
  for n in `seq 0 $nb`
  do
    # select path with rank n
    cat sent$utt.composed.reverse.n | awk '(NR>'$n' || $1!="0"){print;}' | fstcompile | fstconnect > sent$utt.composed.reverse.$n.fst
    fstprint sent$utt.composed.reverse.$n.fst > sent$utt.composed.reverse.$n
    # compute shortest distance to final states
    fstshortestdistance sent$utt.composed.reverse.$n.fst | \
      awk -v list=sent$utt.composed.reverse.$n 'BEGIN{mincost=1E5; while (getline < list > 0){if (NF==2) final[$1]=$2; if (NF==1) final[$1]=0.00001;}} \
      { if (final[$1]) { cost=$2+final[$1]; if (cost<mincost) {mincost=cost;} };} END {print mincost;}' \
      >> sent$utt.reverse.scores
  done

  # present results
  paste sent$utt.forward.scores sent$utt.reverse.scores | \
    awk '{diff=$1-$2; if ( (diff<0?-diff:diff) > 0.001 ) print NR,$1,$2,"!!!"; else print NR,$1,$2;}'
  # clean up
  rm sent$utt
  rm sent$utt.*
done
