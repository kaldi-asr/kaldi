#!/bin/bash


if [ $# != 2 ]; then
  echo "Usage: make_ctms.sh src-dir decode-dir"
  exit 1;
fi

model=$1/final.mdl
dir=$2
if [ ! -f $model ]; then
  echo "No such file $model";
  exit 1;
fi

wbegin=`grep "#1" data/phones_disambig.txt | awk '{print $2}'`
wend=`grep "#2" data/phones_disambig.txt | awk '{print $2}'`

mkdir -p $dir/ctm
for test in mar87 oct87 feb89 oct89 feb91 sep92; do
  ali-to-phones $model ark:$dir/test_${test}.ali ark:- | \
    phones-to-prons data/L_align.fst $wbegin $wend ark:- ark:$dir/test_${test}.tra ark,t:- | \
    prons-to-wordali ark:- \
      "ark:ali-to-phones --write-lengths $model ark:$dir/test_${test}.ali ark:-|" ark,t:- | \
   scripts/wali_to_ctm.sh - data/words.txt > $dir/ctm/test_${test}.ctm || exit 1;
done  


