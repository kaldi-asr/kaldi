#!/bin/bash

# This script basically calls the supplied decoding script
# once for each test set (in parallel on the same machine),
# and then averages the resulting WERs.

mono_opt=

if [ "$1" == "--mono" ]; then
   mono_opt=$1;
   shift;
fi

script=$1
decode_dir=$2 # e.g. exp/sgmm3b/decode
dir=`dirname $decode_dir` # e.g. exp/sgmm3b

if [ $# -lt 2 -o $# -gt 3 ]; then 
  echo "Usage: scripts/decode.sh <decode-script> <decode-dir> [<old-decode-dir>]"
  exit 1;
fi
if [ ! -x $script -o ! -d $dir ]; then
  echo "scripts/decode.sh: Either no such script $script or not exebutable, or no such dir $dir"
  exit 1;
fi

mkdir -p $decode_dir
scripts/mkgraph.sh $mono_opt data/lang_test $dir $dir/graph

if [ $# -eq 2 ]; then # normal case: 2 args.
  for test in mar87 oct87 feb89 oct89 feb91 sep92; do
    $script $dir data/test_$test data/lang $decode_dir/$test &
  done
else
  olddir=$3
  for test in mar87 oct87 feb89 oct89 feb91 sep92; do
    if [ ! -d $olddir/$test ]; then
      echo "decode.sh: no such directory $olddir/$test";
      exit 1;
    fi
    $script $dir data/test_$test data/lang $decode_dir/$test $olddir/$test &
  done
fi
wait
# Average the WERs... there may be various wer files named e.g. wer, wer_10, etc.,
# so do this for each one.
for w in $decode_dir/mar87/wer*; do
  wername=`basename $w`
  scripts/average_wer.sh $decode_dir/?????/$wername > $decode_dir/$wername
done
grep WER $decode_dir/wer* || echo "Error decoding $decode_dir: no WER results found."




