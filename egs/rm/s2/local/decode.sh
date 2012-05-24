#!/bin/bash

script=$1
dir=$2

if [ $# -ne 2 -o ! -x ${script%% *} -o ! -d $dir ]; then
  echo "Usage: scripts/decode.sh <decode-script> <decode-dir>"
  echo "[check your command line arguments]"
fi

scripts/mkgraph.sh data/lang_test $dir $dir/graph

NCPU=$(grep processor /proc/cpuinfo | wc -l)
if [ $NCPU -gt 5 ]; then
  for test in mar87 oct87 feb89 oct89 feb91 sep92; do
    $script $dir data/test_$test data/lang $dir/decode_$test &
  done
  wait
else
  for test in mar87 oct87 feb89 oct89 feb91 sep92; do
    $script $dir data/test_$test data/lang $dir/decode_$test
  done
fi
scripts/average_wer.sh $dir/decode_?????/wer > $dir/wer
cat $dir/wer

