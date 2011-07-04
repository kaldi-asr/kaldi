#!/bin/bash

# Copyright 2010-2011  Microsoft Corporation,  Arnab Ghoshal

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
# (4) get GMM posteriors from this decoded output and estimate fMLLR transforms
#     with this final model
# (5) decode with the final model using both the speaker vectors and fMLLR

if [ -f path.sh ]; then . path.sh; fi
dir=exp/decode_sgmmc_fmllr_norm
tree=exp/sgmmc/tree
occs=exp/sgmmc/final.occs
modelin=exp/sgmmc/final.mdl
alimodelin=exp/sgmmc/final.alimdl
model=exp/sgmmc/final.mdl.norm
alimodel=exp/sgmmc/final.alimdl.norm
fmllr_model=exp/sgmmc/final_fmllr.mdl.norm
graphdir=exp/graph_sgmmc
silphonelist=`cat data/silphones.csl`
preselectmap=exp/ubmb/preselect.map

mincount=1000  # min occupancy to extimate fMLLR transform
iters=10       # number of iters of fMLLR estimation


mkdir -p $dir

sgmm-normalize $modelin ark:$preselectmap $model 2>$dir/normalize.log
sgmm-normalize $alimodelin ark:$preselectmap $alimodel 2>>$dir/normalize.log

sgmm-comp-prexform $model $occs $fmllr_model 2>$dir/prexform.log


scripts/mkgraph.sh $tree $model $graphdir

for test in mar87 oct87 feb89 oct89 feb91 sep92; do
 (
  feats="ark:add-deltas --print-args=false scp:data/test_${test}.scp ark:- |"
  spk2utt_opt="--spk2utt=ark:data/test_${test}.spk2utt"
  utt2spk_opt="--utt2spk=ark:data/test_${test}.utt2spk"
  scripts/compose_maps.pl data/test_${test}.utt2spk data/spk2gender.map | \
   scripts/compose_maps.pl - $preselectmap | \
   gzip -c > $dir/preselect_${test}.gz

  sgmm-gselect "--preselect=ark:gunzip -c $dir/preselect_${test}.gz|" \
    $model "$feats" ark,t:- 2>$dir/gselect.log | \
    gzip -c > $dir/${test}_gselect.gz || exit 1;
  gselect_opt="--gselect=ark:gunzip -c $dir/${test}_gselect.gz|"

  # Use smaller beam for the first pass decoding.
  sgmm-decode-faster "$gselect_opt" --beam=15.0 --acoustic-scale=0.1 --word-symbol-table=data/words.txt $alimodel $graphdir/HCLG.fst "$feats" ark,t:$dir/test_${test}.pass1.tra ark,t:$dir/test_${test}.pass1.ali  2> $dir/pass1_${test}.log

  # Estimate the speaker vectors
  ( ali-to-post ark:$dir/test_${test}.pass1.ali ark:- | \
    weight-silence-post 0.01 $silphonelist $alimodel ark:- ark:- | \
    sgmm-post-to-gpost "$gselect_opt" $alimodel "$feats" ark,s,cs:- ark:- | \
    sgmm-est-spkvecs-gpost $spk2utt_opt $model "$feats" ark,s,cs:- \
    ark:$dir/test_${test}.vecs1 ) 2>$dir/vecs1_${test}.log

  ( ali-to-post ark:$dir/test_${test}.pass1.ali ark:- | \
    weight-silence-post 0.01 $silphonelist $alimodel ark:- ark:- | \
    sgmm-est-spkvecs "$gselect_opt" --spk-vecs=ark:$dir/test_${test}.vecs1 $spk2utt_opt \
      $model "$feats" ark,s,cs:- ark:$dir/test_${test}.vecs2 ) 2>$dir/vecs2_${test}.log

  # Second-pass decoding with the speaker vectors.
  sgmm-decode-faster "$gselect_opt" $utt2spk_opt --spk-vecs=ark:$dir/test_${test}.vecs2 --beam=20.0 --acoustic-scale=0.1 --word-symbol-table=data/words.txt $model $graphdir/HCLG.fst "$feats" ark,t:$dir/test_${test}.pass2.tra ark,t:$dir/test_${test}.pass2.ali  2> $dir/pass2_${test}.log

  # Estimate the fMLLR transforms.
  ( ali-to-post ark:$dir/test_${test}.pass2.ali ark:- | \
    weight-silence-post 0.01 $silphonelist $model ark:- ark:- | \
    sgmm-post-to-gpost --spk-vecs=ark:$dir/test_${test}.vecs2 $utt2spk_opt \
                     "$gselect_opt" $model "$feats" ark,s,cs:- ark:- | \
    sgmm-est-fmllr-gpost --fmllr-iters=$iters --fmllr-min-count=$mincount \
      --spk-vecs=ark:$dir/test_${test}.vecs2 "$spk2utt_opt" $fmllr_model \
      "$feats" ark,s,cs:- ark:$dir/test_${test}.fmllr ) \
      2>$dir/est_fmllr_${test}.log

  adapt_feats="ark:add-deltas --print-args=false scp:data/test_${test}.scp ark:- | transform-feats $utt2spk_opt ark:$dir/test_${test}.fmllr ark:- ark:- |"

  # Now decode with fMLLR-adapted features. Gaussian selection is also done 
  # with the adapted features. This causes a small improvement in WER on RM. 
  sgmm-decode-faster "$gselect_opt" $utt2spk_opt --beam=20.0 --acoustic-scale=0.1 --word-symbol-table=data/words.txt --spk-vecs=ark:$dir/test_${test}.vecs2 $fmllr_model $graphdir/HCLG.fst "$adapt_feats" ark,t:$dir/test_${test}.tra ark,t:$dir/test_${test}.ali  2> $dir/decode_${test}.log

  # the ,p option lets it score partial output without dying..
  scripts/sym2int.pl --ignore-first-field data/words.txt data_prep/test_${test}_trans.txt | \
  compute-wer --mode=present ark:-  ark,p:$dir/test_${test}.tra >& $dir/wer_${test}
 ) &
done

wait

grep WER $dir/wer_* | \
  awk '{n=n+$4; d=d+$6} END{ printf("Average WER is %f (%d / %d) \n", 100.0*n/d, n, d); }' \
   > $dir/wer
