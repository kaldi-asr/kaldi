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

# to be run from ..

if [ -f path.sh ]; then . path.sh; fi
dir=exp/decode_tri2g_diag_utt
mkdir -p $dir
model=exp/tri2g/final.mdl
alignmodel=exp/tri2g/final.alimdl
lvtln=exp/tri2g/final.lvtln
tree=exp/tri2g/tree
graphdir=exp/graph_tri2g
silphones=`cat data/silphones.csl`

# already made the graph.
scripts/mkgraph.sh $tree $model $graphdir

for test in mar87 oct87 feb89 oct89 feb91 sep92; do
 (
  sifeats="ark:add-deltas scp:data/test_${test}.scp ark:- |"

  # First do SI decoding with alignment model.
  # Use smaller beam for this, as less critical.
  gmm-decode-faster --beam=15.0 --acoustic-scale=0.083333 --word-symbol-table=data/words.txt $alignmodel $graphdir/HCLG.fst "$sifeats" ark,t:$dir/test_${test}_pre.tra ark,t:$dir/test_${test}_pre.ali  2> $dir/predecode_${test}.log

  # Comment the two lines below to make this per-utterance.
  #spk2utt_opt=--spk2utt=ark:data/test_${test}.spk2utt
  #utt2spk_opt=--utt2spk=ark:data/test_${test}.utt2spk
  
 ( ali-to-post ark:$dir/test_${test}_pre.ali ark:- | \
    weight-silence-post 0.0 $silphones $alignmodel ark:- ark:- | \
    gmm-post-to-gpost $alignmodel "$sifeats" ark:- ark:- | \
    gmm-est-lvtln-trans --norm-type=diag --verbose=1 $spk2utt_opt  $model $lvtln \
      "$sifeats" ark:- ark:$dir/lvtln_${test}.trans ark,t:$dir/lvtln_${test}.warp ) \
     2>$dir/lvtln_${test}.log || exit 1;

  feats="ark:add-deltas scp:data/test_${test}.scp ark:- | transform-feats $utt2spk_opt ark:$dir/lvtln_${test}.trans ark:- ark:- |"

  gmm-decode-faster --beam=20.0 --acoustic-scale=0.083333 --word-symbol-table=data/words.txt $model $graphdir/HCLG.fst "$feats" ark,t:$dir/test_${test}.tra ark,t:$dir/test_${test}.ali  2> $dir/decode_${test}.log

 # the ,p option lets it score partial output without dying..
  scripts/sym2int.pl --ignore-first-field data/words.txt data_prep/test_${test}_trans.txt | \
  compute-wer --mode=present ark:-  ark,p:$dir/test_${test}.tra >& $dir/wer_${test}
 ) &
done

wait

grep WER $dir/wer_* | \
  awk '{n=n+$4; d=d+$6} END{ printf("Average WER is %f (%d / %d) \n", 100.0*n/d, n, d); }' \
   > $dir/wer
