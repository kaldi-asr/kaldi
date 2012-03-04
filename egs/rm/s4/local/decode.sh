#!/bin/bash

# This script basically calls the supplied decoding script
# once for each test set (in parallel on the same machine),
# and then averages the resulting WERs.
# The interpretation of the decode-dir-1, etc., as inputs,
# outputs and so on, depends on the decoding script you call.

# It assumes the model directory is one level of from decode-dir-1.

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

if [ $# -ne 2 ]; then 
  echo "Usage: scripts/decode.sh <decode-script> <decode-dir-1>"
  exit 1;
fi
if [ ! -x $script -o ! -d $dir ]; then
  echo "scripts/decode.sh: Either no such script $script or not executable, or no such dir $dir"
  exit 1;
fi

scripts/mkgraph.sh $mono_opt data/lang_test $dir $dir/graph

$script $dir data/test data/lang $decode_dir_1/ &
wait

# The publicly available RM subset has just one test set(instead of mar87 etc.),
# so no averaging is needed
grep WER $decode_dir_1/wer* || echo "Error decoding $decode_dir: no WER results found."
