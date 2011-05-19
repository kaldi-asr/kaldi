
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

# deocde_tri_fmllr.sh is as decode_tri.sh but estimating fMLLR in test,
# per speaker.  There is no SAT.
# To be run from ..

if [ -f path.sh ]; then . path.sh; fi
srcdir=exp/decode_tri2a
dir=exp/decode_tri2a_fmllr
mkdir -p $dir
model=exp/tri2a/final.mdl
tree=exp/tri2a/tree
graphdir=exp/graph_tri2a
silphones=`cat data/silphones.csl`

mincount=500 # mincount before we estimate a transform.

scripts/mkgraph.sh $tree $model $graphdir

for test in mar87 oct87 feb89 oct89 feb91 sep92; do
 (
  # Comment the two lines below to make this per-utterance.
  # This would only work if $srcdir was also per-utterance [otherwise
  # you'd have to mess with the script a bit].
  spk2utt_opt=--spk2utt=ark:data/test_${test}.spk2utt
  utt2spk_opt=--utt2spk=ark:data/test_${test}.utt2spk

  sifeats="ark:add-deltas --print-args=false scp:data/test_${test}.scp ark:- |"

  ali-to-post ark:$srcdir/test_${test}.ali ark:- | \
    weight-silence-post 0.01 $silphones $model ark:- ark:- | \
    gmm-est-fmllr --fmllr-min-count=$mincount $spk2utt_opt $model \
     "$sifeats" ark,o:- ark:$dir/${test}.fmllr 2>$dir/fmllr_${test}.log

  feats="ark:add-deltas --print-args=false scp:data/test_${test}.scp ark:- | transform-feats $utt2spk_opt ark:$dir/${test}.fmllr ark:- ark:- |"

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

