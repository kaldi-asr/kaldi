#!/bin/bash

script=$1
dir=$2

if [ $# -lt 2 -o $# -gt 3 -o ! -x $script -o ! -d $dir ]; then
  echo "Usage: scripts/decode.sh <decode-script> <decode-dir> [<old-decode-dir>]"
  echo "[check your command line arguments]"
fi

scripts/mkgraph.sh data/lang_test $dir $dir/graph

if [ $# -eq 2 ]; then # normal case: 2 args.
  for test in mar87 oct87 feb89 oct89 feb91 sep92; do
    $script $dir data/test_$test data/lang $dir/decode_$test &
  done
else
  olddir=$3
  for test in mar87 oct87 feb89 oct89 feb91 sep92; do
    if [ ! -d $olddir/decode_$test ]; then
      echo "decode.sh: no such directory $oldir/decode_$test";
      exit 1;
    fi
    $script $dir data/test_$test data/lang $dir/decode_$test $olddir/decode_$test &
  done
fi
wait
scripts/average_wer.sh $dir/decode_?????/wer > $dir/wer
cat $dir/wer

