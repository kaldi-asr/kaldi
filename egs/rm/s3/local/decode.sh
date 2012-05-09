#!/bin/bash

# This script basically calls the supplied decoding script
# once for each test set (in parallel on the same machine),
# and then averages the resulting WERs.
# The interpretation of the decode-dir-1, etc., as inputs,
# outputs and so on, depends on the decoding script you call.

# It assumes the model directory is one level up from decode-dir-1.

mono_opt=

if [ "$1" == "--mono" ]; then
   mono_opt=$1;
   shift;
fi

script=$1
decode_dir_1=$2 # e.g. exp/sgmm3b/decode
decode_dir_2=$3
decode_dir_3=$4
dir=`dirname $decode_dir_1` # e.g. exp/sgmm3b

if [ $# -lt 2 -o $# -gt 4 ]; then 
  echo "Usage: scripts/decode.sh <decode-script> <decode-dir-1> [<decode-dir-2> [<decode-dir-3>] ]"
  exit 1;
fi
if [ ! -x $script -o ! -d $dir ]; then
  echo "scripts/decode.sh: Either no such script $script or not executable, or no such dir $dir"
  exit 1;
fi

scripts/mkgraph.sh $mono_opt data/lang_test $dir $dir/graph

if [ $# -eq 2 ]; then # normal case: 2 args.
  for test in mar87 oct87 feb89 oct89 feb91 sep92; do
    $script $dir data/test_$test data/lang $decode_dir_1/$test &
  done
elif [ $# -eq 3 ]; then
  for test in mar87 oct87 feb89 oct89 feb91 sep92; do
    $script $dir data/test_$test data/lang $decode_dir_1/$test $decode_dir_2/$test &
  done
else
  for test in mar87 oct87 feb89 oct89 feb91 sep92; do
    $script $dir data/test_$test data/lang $decode_dir_1/$test $decode_dir_2/$test $decode_dir_3/$test &
  done
fi
wait



# Average the WERs... there may be various wer files named e.g. wer, wer_10, etc.,
# so do this for each one.
for w in $decode_dir_1/mar87/wer*; do
  wername=`basename $w`
  scripts/average_wer.sh $decode_dir_1/?????/$wername > $decode_dir_1/$wername
done
grep WER $decode_dir_1/wer* || echo "Error decoding $decode_dir: no WER results found."

