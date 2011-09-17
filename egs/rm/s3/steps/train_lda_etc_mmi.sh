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

# To be run from ..
# This directory does MMI model training, starting from trained
# models.  The models must be trained on raw features plus
# cepstral mean normalization plus splice-9-frames, an LDA+[something] 
# transform, then possibly speaker-specific affine transforms 
# (fMLLR/CMLLR).   This script works out from the alignment directory
# whether you trained with some kind of speaker-specific transform.
#
# This training run starts from an initial directory that has
# alignments, models and transforms from an LDA+MLLT system:
#  ali, final.mdl, final.mat


if [ $# != 4 ]; then
   echo "Usage: steps/train_lda_etc_mmi.sh <data-dir> <lang-dir> <ali-dir> <exp-dir>"
   echo " e.g.: steps/train_lda_etc_mmi.sh data/train data/lang exp/tri3d_ali exp/tri4a"
   exit 1;
fi

if [ -f path.sh ]; then . path.sh; fi

data=$1
lang=$2
alidir=$3
dir=$4

num_iters=4
acwt=0.1
beam=20
latticebeam=10
scale_opts="--transition-scale=1.0 --acoustic-scale=0.1 --self-loop-scale=0.1"

mkdir -p $dir
cp $alidir/tree $alidir/final.mat $dir # Will use the same tree and transforms as in the baseline.
cp $alidir/final.mdl $dir/0.mdl

if [ -f $alidir/final.alimdl ]; then
   cp $alidir/final.alimdl $dir/final.alimdl
   cp $alidir/final.mdl $dir/final.adaptmdl # This model used by decoding scripts,
   # when you don't want to compute fMLLR transforms with the MMI-trained model.
fi

scripts/split_scp.pl $data/feats.scp $dir/feats{0,1,2,3}.scp

feats="ark:apply-cmvn --norm-vars=false --utt2spk=ark:$data/utt2spk ark:$alidir/cmvn.ark scp:$data/feats.scp ark:- | splice-feats ark:- ark:- | transform-feats $dir/final.mat ark:- ark:- |"
for n in 0 1 2 3; do
  featspart[$n]="ark:apply-cmvn --norm-vars=false --utt2spk=ark:$data/utt2spk ark:$alidir/cmvn.ark scp:$dir/feats$n.scp ark:- | splice-feats ark:- ark:- | transform-feats $dir/final.mat ark:- ark:- |"
done

if [ -f $alidir/trans.ark ]; then
   echo "Running with speaker transforms $alidir/trans.ark"
   feats="$feats transform-feats --utt2spk=ark:$data/utt2spk ark:$alidir/trans.ark ark:- ark:- |"
   for n in 0 1 2 3; do
     featspart[$n]="${featspart[$n]} transform-feats --utt2spk=ark:$data/utt2spk ark:$alidir/trans.ark ark:- ark:- |"
   done
fi

# compute integer form of transcripts.
scripts/sym2int.pl --ignore-first-field $lang/words.txt < $data/text > $dir/train.tra \
  || exit 1;

cp -r $lang $dir/lang

# Compute grammar FST which corresponds to unigram decoding graph.
cat $dir/train.tra | awk '{for(n=2;n<=NF;n++){ printf("%s ", $n); } printf("\n"); }' | \
 scripts/make_unigram_grammar.pl | fstcompile > $dir/lang/G.fst \
  || exit 1;

# mkgraph.sh expects a whole directory "lang", so put everything in one directory...
# it gets L_disambig.fst and G.fst (among other things) from $dir/lang, and 
# final.mdl from $alidir; the output HCLG.fst goes in $dir/graph.

scripts/mkgraph.sh $dir/lang $alidir $dir/dgraph || exit 1;

echo "Making denominator lattices"


rm $dir/.error 2>/dev/null
for n in 0 1 2 3; do
   gmm-latgen-simple --beam=$beam --lattice-beam=$latticebeam --acoustic-scale=$acwt \
    --word-symbol-table=$lang/words.txt \
    $alidir/final.mdl $dir/dgraph/HCLG.fst "${featspart[$n]}" "ark:|gzip -c >$dir/lat$n.gz" \
     2>$dir/decode_den.$n.log || touch $dir/.error &
done
wait
if [ -f $dir/.error ]; then
   echo "Error creating denominator lattices"
   exit 1;
fi

# No need to create "numerator" alignments/lattices: we just use the 
# alignments in $alidir.

echo "Note: ignore absolute offsets in the objective function values"
echo "This is caused by not having LM, lexicon or transition-probs in numerator"

x=0;
while [ $x -lt $num_iters ]; do
  echo "Iteration $x: getting denominator stats."
  # Get denominator stats...
  if [ $x -eq 0 ]; then
    ( lattice-to-post --acoustic-scale=$acwt "ark:gunzip -c $dir/lat?.gz|" ark:- | \
      gmm-acc-stats $dir/$x.mdl "$feats" ark:- $dir/den_acc.$x.acc ) \
     2>$dir/acc_den.$x.log || exit 1;
  else # Need to recompute acoustic likelihoods...
   ( gmm-rescore-lattice $dir/$x.mdl "ark:gunzip -c $dir/lat?.gz|" "$feats" ark:- | \
      lattice-to-post --acoustic-scale=$acwt ark:- ark:- | \
      gmm-acc-stats $dir/$x.mdl "$feats" ark:- $dir/den_acc.$x.acc ) \
     2>$dir/acc_den.$x.log || exit 1;
  fi
  echo "Iteration $x: getting numerator stats."
  # Get numerator stats...
  gmm-acc-stats-ali $dir/$x.mdl "$feats" ark:$alidir/ali $dir/num_acc.$x.acc \
   2>$dir/acc_num.$x.log || exit 1;
  # Update.
  gmm-est-mmi $dir/$x.mdl $dir/num_acc.$x.acc $dir/den_acc.$x.acc $dir/$[$x+1].mdl \
    2>$dir/update.$x.log || exit 1;

  den=`grep Overall $dir/acc_den.$x.log  | grep lattice-to-post | awk '{print $7}'`
  num=`grep Overall $dir/acc_num.$x.log  | grep gmm-acc-stats-ali | awk '{print $11}'`
  diff=`perl -e "print ($num * $acwt - $den);"`
  impr=`grep Overall $dir/update.$x.log | awk '{print $10;}'`
  impr=`perl -e "print ($impr * $acwt);"` # auxf impr normalized by multiplying by
  # kappa, so it's comparable to an objective-function change.
  echo On iter $x, objf was $diff, auxf improvement was $impr | tee $dir/objf.$x.log

  x=$[$x+1]
done

# Just copy the source-dir's occs, in case we later need them for something...
cp $alidir/final.occs $dir
( cd $dir; ln -s $x.mdl final.mdl )


echo Done

