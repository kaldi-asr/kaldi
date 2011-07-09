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

# SGMM decoding with adaptation.
# 
# SGMM decoding; use a different acoustic scale from normal (0.1 vs 0.08333)
# (1) decode with "alignment model"
# (2) get GMM posteriors with "alignment model" and estimate speaker
#     vectors with final model
# (3) decode with final model.

if [ -f path.sh ]; then . path.sh; fi
dir=exp/decode_sgmme
tree=exp/sgmme/tree
model=exp/sgmme/final.mdl
alimodel=exp/sgmme/final.alimdl
graphdir=exp/graph_sgmme
silphonelist=`cat data/silphones.csl`
mat=exp/tri2k/lda.mat
# see below, we also use exp/decode_tri2k
mkdir -p $dir

scripts/mkgraph.sh $tree $model $graphdir

for test in mar87 oct87 feb89 oct89 feb91 sep92; do
 (
  trans=exp/decode_tri2k/et_${test}.trans
  spk2utt_opt="--spk2utt=ark:data/test_${test}.spk2utt"
  utt2spk_opt="--utt2spk=ark:data/test_${test}.utt2spk"
  feats="ark:splice-feats scp:data/test_${test}.scp ark:- | transform-feats $mat ark:- ark:- | transform-feats $utt2spk_opt ark:$trans ark:- ark:- |"

  sgmm-gselect $model "$feats" ark,t:- 2>$dir/gselect.log | \
     gzip -c > $dir/${test}_gselect.gz || exit 1;
  gselect_opt="--gselect=ark:gunzip -c $dir/${test}_gselect.gz|"

  # Use smaller beam first time.
  sgmm-decode-faster "$gselect_opt" --beam=15.0 --acoustic-scale=0.1 --word-symbol-table=data/words.txt $alimodel $graphdir/HCLG.fst "$feats" ark,t:$dir/test_${test}.pre_tra ark,t:$dir/test_${test}.pre_ali  2> $dir/predecode_${test}.log

  ( ali-to-post ark:$dir/test_${test}.pre_ali ark:- | \
    weight-silence-post 0.01 $silphonelist $alimodel ark:- ark:- | \
    sgmm-post-to-gpost "$gselect_opt" $alimodel "$feats" ark,s,cs:- ark:- | \
    sgmm-est-spkvecs-gpost $spk2utt_opt $model "$feats" ark,s,cs:- \
       ark:$dir/test_${test}.vecs1 ) 2>$dir/vecs1_${test}.log

  ( ali-to-post ark:$dir/test_${test}.pre_ali ark:- | \
    weight-silence-post 0.01 $silphonelist $alimodel ark:- ark:- | \
    sgmm-est-spkvecs --spk-vecs=ark:$dir/test_${test}.vecs1 $spk2utt_opt \
      $model "$feats" ark,s,cs:- ark:$dir/test_${test}.vecs2 ) 2>$dir/vecs2_${test}.log
  

  sgmm-decode-faster "$gselect_opt" $utt2spk_opt --spk-vecs=ark:$dir/test_${test}.vecs2 --beam=20.0 --acoustic-scale=0.1 --word-symbol-table=data/words.txt $model $graphdir/HCLG.fst "$feats" ark,t:$dir/test_${test}.tra ark,t:$dir/test_${test}.ali  2> $dir/decode_${test}.log

 # the ,p option lets it score partial output without dying..
  scripts/sym2int.pl --ignore-first-field data/words.txt data_prep/test_${test}_trans.txt | \
  compute-wer --mode=present ark:-  ark,p:$dir/test_${test}.tra >& $dir/wer_${test}
 ) &
done

wait

grep WER $dir/wer_* | \
  awk '{n=n+$4; d=d+$6} END{ printf("Average WER is %f (%d / %d) \n", 100.0*n/d, n, d); }' \
   > $dir/wer
