#!/bin/bash

# Copyright 2010-2011 Microsoft Corporation

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

# Decode the testing data.

if [ -f path.sh ]; then . path.sh; fi
dir=exp/decode_tri2c
mkdir -p $dir
model=exp/tri2c/final.mdl
tree=exp/tri2c/tree
graphdir=exp/graph_tri2c
# Note, the following 3 options must match the same options in train_tri2c.sh
norm_vars=false
after_deltas=false
per_spk=true

scripts/mkgraph.sh $tree $model $graphdir

for test in mar87 oct87 feb89 oct89 feb91 sep92; do
 (
  if [ $per_spk == "true" ]; then
    spk2utt_opt=--spk2utt=ark:data/test_${test}.spk2utt
    utt2spk_opt=--utt2spk=ark:data/test_${test}.utt2spk
  fi # else empty.

  echo "Computing cepstral mean and variance stats."
  # compute mean and variance stats.
  if [ $after_deltas == true ]; then
    add-deltas --print-args=false scp:data/test_${test}.scp ark:- | compute-cmvn-stats $spk2utt_opt ark:- ark:$dir/cmvn_${test}ark 2>$dir/cmvn_${test}.log
    feats="ark:add-deltas --print-args=false scp:data/test_${test}.scp ark:- | apply-cmvn --norm-vars=$norm_vars $utt2spk_opt ark:$dir/cmvn_${test}ark ark:- ark:- |"
  else 
    compute-cmvn-stats --spk2utt=ark:data/test_${test}.spk2utt scp:data/test_${test}.scp ark:$dir/cmvn_${test} 2>$dir/cmvn_${test}.log
    feats="ark:apply-cmvn --norm-vars=$norm_vars $utt2spk_opt ark:$dir/cmvn_${test} scp:data/test_${test}.scp ark:- | add-deltas --print-args=false ark:- ark:- |"
  fi


  gmm-decode-faster --beam=20.0 --acoustic-scale=0.08333 --word-symbol-table=data/words.txt $model $graphdir/HCLG.fst "$feats" ark,t:$dir/test_${test}.tra ark,t:$dir/test_${test}.ali  2> $dir/decode_${test}.log

# the ,p option lets it score partial output without dying..

  scripts/sym2int.pl --ignore-first-field data/words.txt data_prep/test_${test}_trans.txt | \
  compute-wer --mode=present ark:-  ark,p:$dir/test_${test}.tra > $dir/wer_${test}
) &
done
wait

grep WER $dir/wer_* | \
  awk '{n=n+$4; d=d+$6} END{ printf("Average WER is %f (%d / %d) \n", 100.0*n/d, n, d); }' \
   > $dir/wer
