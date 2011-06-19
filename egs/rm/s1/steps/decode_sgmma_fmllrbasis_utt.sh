#!/bin/bash

# Copyright 2010-2011  Arnab Ghoshal,  Microsoft Corporation

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

# SGMM decoding with fMLLR adaptation. The steps are:
# (1) decode with the SI (unadapted) model
# (2) get GMM posteriors from this decoded output and estimate fMLLR transforms
# (3) decode again using the fMLLR transforms

if [ -f path.sh ]; then . path.sh; fi
dir=exp/decode_sgmma_fmllrbasis_utt
tree=exp/sgmma/tree
model=exp/sgmma/final_fmllrbasis.mdl
occs=exp/sgmma/final.occs
graphdir=exp/graph_sgmma
silphonelist=`cat data/silphones.csl`

mincount=100  # min occupancy to extimate fMLLR transform
iters=10      # number of iters of fMLLR estimation
scale=0.2     # Scale occupancies to determine number of fMLLR bases to use
nbases=200    # Use at most these many fMLLR bases 

if [ ! -f $model ]; then
  echo "Cannot find $model. Maybe the fMLLR bases weren't estiamted?"
  exit 1;
fi

mkdir -p $dir

scripts/mkgraph.sh $tree $model $graphdir

for test in mar87 oct87 feb89 oct89 feb91 sep92; do
 (
  feats="ark:add-deltas --print-args=false scp:data/test_${test}.scp ark:- |"

  sgmm-gselect $model "$feats" ark,t:- 2>$dir/gselect.log | \
     gzip -c > $dir/${test}_gselect.gz || exit 1;
  gselect_opt="--gselect=ark:gunzip -c $dir/${test}_gselect.gz|"

  # Use smaller beam for the first pass decoding.
  sgmm-decode-faster "$gselect_opt" --beam=15.0 --acoustic-scale=0.1 --word-symbol-table=data/words.txt $model $graphdir/HCLG.fst "$feats" ark,t:$dir/test_${test}.pre_tra ark,t:$dir/test_${test}.pre_ali 2> $dir/predecode_${test}.log

  # Estimate the fMLLR transforms.
  ( ali-to-post ark:$dir/test_${test}.pre_ali ark:- | \
    weight-silence-post 0.0 $silphonelist $model ark:- ark:- | \
    sgmm-post-to-gpost "$gselect_opt" $model "$feats" ark,s,cs:- ark:- | \
    sgmm-est-fmllr-gpost --fmllr-iters=$iters --fmllr-bases-occ-scale=$scale \
      --fmllr-min-count-bases=$mincount --fmllr-num-bases=$nbases --verbose=3 \
      $model "$feats" ark,s,cs:- ark:$dir/test_${test}.fmllr ) \
      2>$dir/est_fmllr_${test}.log

  adapt_feats="ark:add-deltas --print-args=false scp:data/test_${test}.scp ark:- | transform-feats $utt2spk_opt ark:$dir/test_${test}.fmllr ark:- ark:- |"

  # Now decode with fMLLR-adapted features. Gaussian selection is also done 
  # with the adapted features. This causes a small improvement in WER on RM. 
  sgmm-decode-faster --beam=20.0 --acoustic-scale=0.1 --word-symbol-table=data/words.txt $model $graphdir/HCLG.fst "$adapt_feats" ark,t:$dir/test_${test}.tra ark,t:$dir/test_${test}.ali 2> $dir/decode_${test}.log

  # the ,p option lets it score partial output without dying..
  scripts/sym2int.pl --ignore-first-field data/words.txt data_prep/test_${test}_trans.txt | \
  compute-wer --mode=present ark:-  ark,p:$dir/test_${test}.tra >& $dir/wer_${test}
 ) &
done

wait

grep WER $dir/wer_* | \
  awk '{n=n+$4; d=d+$6} END{ printf("Average WER is %f (%d / %d) \n", 100.0*n/d, n, d); }' \
   > $dir/wer

