
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

# SGMM decoding; use a different acoustic scale from normal (0.1 vs 0.08333)

if [ -f path.sh ]; then . path.sh; fi
dir=exp/decode_sgmm2
tree=exp/sgmm/tree
model=exp/sgmm2/final.mdl
graphdir=exp/graph_sgmm2

mkdir -p $dir

scripts/mkgraph.sh $tree $model $graphdir

for test in mar87 oct87 feb89 oct89 feb91 sep92; do
 (
  feats="ark:add-deltas --print-args=false scp:data/test_${test}.scp ark:- |"

  sgmm-gselect $model "$feats" ark,t:- 2>$dir/gselect_${test}.log | gzip -c > $dir/gselect_${test}.gz || exit 1;
  gselect_opt="--gselect-read=ark:gunzip -c $dir/gselect_${test}.gz|"
  sgmm-decode-faster-spkvecs --beam=20.0 --acoustic-scale=0.1 --word-symbol-table=data/words.txt "$gselect_opt" $model $graphdir/HCLG.fst "$feats" ark,t:$dir/test_${test}.tra ark,t:$dir/test_${test}.ali  2> $dir/decode_${test}.log || exit 1;
  ali-to-post $dir/test_${test}.ali $dir/test_${test}.post 2> $dir/post_${test}.log || exit 1;

  gselect_opt="--gselect=ark:gunzip -c $dir/gselect_${test}.gz|"
  spk2utt_opt=--spk2utt=ark:data/test_${test}.spk2utt
  utt2spk_opt=--utt2spk=ark:data/test_${test}.utt2spk
  sgmm-est-spkvecs "$gselect_opt" --spk2utt= $model "$feats" $dir/test_${test}.post $dir/vecs_${test} 2> $dir/est_spkvecs_${test}.log || exit 1;
  sgmm-decode-faster-spkvecs --beam=20.0 --acoustic-scale=0.1 --word-symbol-table=data/words.txt "$gselect_opt" --spkvecs-read=$dir/vecs_${test} $model $graphdir/HCLG.fst "$feats" ark,t:$dir/test_${test}.tra ark,t:$dir/test_${test}.ali  2> $dir/decode_vecs_${test}.log

 # the ,p option lets it score partial output without dying..
  scripts/sym2int.pl --ignore-first-field data/words.txt data_prep/test_${test}_trans.txt | \
  compute-wer --mode=present ark:-  ark,p:$dir/test_${test}.tra >& $dir/wer_${test}
 ) &
done

wait

cat $dir/wer_* | \
  awk '{n=n+$4; d=d+$6} END{ printf("Average WER is %f (%d / %d) \n", 100.0*n/d, n, d); }' \
   > $dir/wer
