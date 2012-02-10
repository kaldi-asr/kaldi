#!/bin/bash

# This script just calls the supplied decoding script
# (typically steps/decode_combine.sh, which is not the
# same as this script) once for each test set,
# and then averages the resulting WERs.


script=$1
decode_dir_in1=$2
decode_dir_in2=$3
decode_dir_out=$4

if [ $# -ne 4 ]; then
  echo "Usage: scripts/decode_combine.sh <decode-script> <decode-dir-in1> <decode-dir-in2> <decode-dir-out>"
  exit 1;
fi
if [ ! -x $script -o ! -d $decode_dir_in1 -o ! -d $decode_dir_in2 ]; then
  echo "scripts/decode.sh: Either no such script $script or not executable, or no such dir $decode_dir_in1 or $decode_dir_in2"
  exit 1;
fi

mkdir -p $decode_dir_out

for test in mar87 oct87 feb89 oct89 feb91 sep92; do
  $script data/test_$test data/lang $decode_dir_in1/$test $decode_dir_in2/$test $decode_dir_out/$test
done
wait
# Average the WERs... there may be various wer files named e.g. wer, wer_10, etc.,
# so do this for each one.
for w in $decode_dir_out/mar87/wer*; do
  wername=`basename $w`
  scripts/average_wer.sh $decode_dir_out/?????/$wername > $decode_dir_out/$wername
done
grep WER $decode_dir_out/wer* || echo "Error decoding $decode_dir_out: no WER results found."
