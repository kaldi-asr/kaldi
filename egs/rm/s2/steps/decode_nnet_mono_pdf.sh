#!/bin/bash

# Copyright 2011 Karel Vesely

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#  http://www.apache.org/licenses/LICENSE-2.0
#
# THIS CODE IS PROVIDED *AS IS* BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, EITHER EXPRESS OR IMPLIED, INCLUDING WITHOUT LIMITATION ANY IMPLIED
# WARRANTIES OR CONDITIONS OF TITLE, FITNESS FOR A PARTICULAR PURPOSE,
# MERCHANTABLITY OR NON-INFRINGEMENT.
# See the Apache 2 License for the specific language governing permissions and
# limitations under the License.

# Pure hybrid monophone decoding script.


if [ -f path.sh ]; then . path.sh; fi

acousticscale=0.22

dir=exp/decode_nnet_mono_pdf
tree=exp/mono/tree
mkdir -p $dir
model=exp/mono/final.mdl
graphdir=exp/graph_mono
nnet=exp/nnet_mono_pdf/nnet.final

scripts/mkgraph.sh --mono $tree $model $graphdir

echo "DECODING..."
for test in mar87 oct87 feb89 oct89 feb91 sep92; do
 (
  #get features
  feats="ark:add-deltas --print-args=false scp:data/test_${test}.scp ark:- |"
  #compute CMVN
  cmvn=ark:$dir/test_${test}_cmvn.ark
  compute-cmvn-stats "$feats" $cmvn
  feats="$feats apply-cmvn --print-args=false --norm-vars=true $cmvn ark:- ark:- |"
  #add MLP transform
  feats="$feats nnet-forward --print-args=false --apply-log=true $nnet ark:- ark:- |"

  echo $feats

  decode-faster-mapped --beam=20.0 --acoustic-scale=$acousticscale --word-symbol-table=data/words.txt $model $graphdir/HCLG.fst "$feats" ark,t:$dir/test_${test}.tra ark,t:$dir/test_${test}.ali  2> $dir/decode_${test}.log

  # the ,p option lets it score partial output without dying..
  scripts/sym2int.pl --ignore-first-field data/words.txt data_prep/test_${test}_trans.txt | \
  compute-wer --mode=present ark:-  ark,p:$dir/test_${test}.tra >& $dir/wer_${test}
 ) &
done

wait

grep WER $dir/wer_* | \
  awk '{n=n+$4; d=d+$6} END{ printf("Average WER is %f (%d / %d) \n", (100.0*n)/d, n, d); }' \
   | tee $dir/wer

