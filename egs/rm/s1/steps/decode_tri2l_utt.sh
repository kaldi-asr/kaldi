
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
dir=exp/decode_tri2l_utt
mkdir -p $dir
model=exp/tri2l/final.mdl
alignmodel=exp/tri2l/final.alimdl
tree=exp/tri2l/tree
graphdir=exp/graph_tri2l 
transform=exp/tri2l/final.mat
silphones=`cat data/silphones.csl`
mincount=300

scripts/mkgraph.sh $tree $model $graphdir

for test in mar87 oct87 feb89 oct89 feb91 sep92; do
 (
  #spk2utt_opt=--spk2utt=ark:data/test_${test}.spk2utt
  #utt2spk_opt=--utt2spk=ark:data/test_${test}.utt2spk

  sifeats="ark:splice-feats scp:data/test_${test}.scp ark:- | transform-feats $transform ark:- ark:-|"

  # Use smaller beam for 1st pass.
  gmm-decode-faster --beam=17.0 --acoustic-scale=0.083333 --word-symbol-table=data/words.txt $alignmodel $graphdir/HCLG.fst "$sifeats" ark,t:$dir/test_${test}.pre_tra ark,t:$dir/test_${test}.pre_ali  2> $dir/predecode_${test}.log

 ( ali-to-post ark:$dir/test_${test}.pre_ali ark:- | \
    weight-silence-post 0.0 $silphones $model ark:- ark:- | \
    gmm-est-fmllr --fmllr-min-count=$mincount $spk2utt_opt $model \
    "$sifeats" ark,o:- ark:$dir/${test}.fmllr ) 2>$dir/fmllr_${test}.log
  
  feats="ark:splice-feats scp:data/test_${test}.scp ark:- | transform-feats $transform ark:- ark:- | transform-feats $utt2spk_opt ark:$dir/${test}.fmllr ark:- ark:- |"

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
