#!/bin/bash
# Copyright 2010-2011 Chao Weng  Microsoft Corporation

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

mce_alpha=0.1 # Constant used in MCE computation
mce_beta=0.0 # ditto; will normally be 0.
tau=100 # Tau value.
stage=-4

if [ $1 == "--stage" ]; then # e.g. "--stage 0"
   shift;
   stage=$1;
   shift;
fi

if [ $# != 4 ]; then
   echo "Usage: steps/train_lda_etc_mce.sh <data-dir> <lang-dir> <ali-dir> <exp-dir>"
   echo " e.g.: steps/train_lda_etc_mce.sh data/train data/lang exp/tri3d_ali exp/tri4a"
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
scale_opts_noac="--transition-scale=1.0 --self-loop-scale=0.1"
scale_opts="$scale_opts_noac --acoustic-scale=0.1"
silphonelist=`cat $lang/silphones.csl`

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


if [ $stage -le -4 ]; then
  cp -r $lang $dir/

  # Compute grammar FST which corresponds to unigram decoding graph.
  cat $dir/train.tra | awk '{for(n=2;n<=NF;n++){ printf("%s ", $n); } printf("\n"); }' | \
   scripts/make_unigram_grammar.pl | fstcompile > $dir/lang/G.fst \
  || exit 1;


  # mkgraph.sh expects a whole directory "lang", so put everything in one directory...
  # it gets L_disambig.fst and G.fst (among other things) from $dir/lang, and 
  # final.mdl from $alidir; the output HCLG.fst goes in $dir/graph.

  scripts/mkgraph.sh $dir/lang $alidir $dir/dengraph || exit 1;
fi

if [ $stage -le -3 ]; then

  echo "Making numerator lattices"

  if [ ! -f $dir/lang/LG.fst ]; then
    fsttablecompose $dir/lang/L.fst $dir/lang/G.fst > $dir/lang/LG.fst || exit 1
  fi

  scripts/split_scp.pl $data/text $dir/text{0,1,2,3}

  rm $dir/.error 2>/dev/null
  for n in 0 1 2 3; do
    tra="ark:scripts/sym2int.pl --ignore-first-field $lang/words.txt < $dir/text$n |"
    # Note: we use gmm-latgen-faster as the command-line interface of gmm-latgen-simple
    # doesn't currently support FSTs as input.
    ( compile-train-graphs $scale_opts_noac $dir/tree $alidir/final.mdl  $dir/lang/LG.fst "$tra" ark:- | \
     gmm-latgen-faster --beam=$beam --lattice-beam=$latticebeam --acoustic-scale=$acwt \
      --word-symbol-table=$lang/words.txt $alidir/final.mdl ark:- "${featspart[$n]}" \
      "ark:|gzip -c >$dir/numlat$n.gz" ) 2>$dir/decode_num.$n.log || touch $dir/.error &
  done
  wait
  [ -f $dir/.error ] && echo "Error creating numerator lattices" && exit 1;
fi

if [ $stage -le -2 ]; then

  echo "Making denominator lattices (minus numerator)"

  for n in 0 1 2 3; do
     gmm-latgen-simple --beam=$beam --lattice-beam=$latticebeam --acoustic-scale=$acwt \
      --word-symbol-table=$lang/words.txt \
      $alidir/final.mdl $dir/dengraph/HCLG.fst "${featspart[$n]}" \
      "ark:|lattice-difference ark:- 'ark:gunzip -c $dir/numlat$n.gz|' 'ark:|gzip -c >$dir/denlat$n.gz'"  2>$dir/decode_den.$n.log || touch $dir/.error &
  done
  wait
  [ -f $dir/.error ] && echo "Error creating denominator lattices" && exit 1;
fi

x=0;
tmpdir=/tmp/tmp.$$
mkdir -p $tmpdir
trap "rm -rf $tmpdir" EXIT HUP KILL

while [ $x -lt $num_iters ]; do
  if [ $stage -le $x ]; then
    # Write lattice posteriors, from num and den, to archive in /tmp
    for type in num den; do
      echo "Iteration $x of MCE: getting $type posts and scores"
      (  gunzip -c $dir/${type}lat?.gz | \
       ( if [ $x -eq 0 ]; then cat; else gmm-rescore-lattice $dir/$x.mdl ark:- "$feats" ark:-; fi ) | \
       lattice-to-post --acoustic-scale=$acwt ark:- ark:- ark,t:$dir/${type}${x}.likes | \
       gzip -c >$tmpdir/${type}.posts.gz ) 2>$dir/${type}_to_post.$x.log || exit 1;
   done

    # Compute MCE scale.
    echo "Computing MCE scaling factor on iteration $x"
    compute-mce-scale --mce-alpha=$mce_alpha --mce-beta=$mce_beta ark:$dir/num$x.likes ark:$dir/den$x.likes ark,t:$dir/post$x.scale \
      2>$dir/compute_mce_scale.$x.log || exit 1;
  
    for type in num den; do
      echo "Iteration $x of MCE: computing $type stats"
      gmm-acc-stats $dir/$x.mdl "$feats" "ark:gunzip -c $tmpdir/${type}.posts.gz | scale-post ark:- ark:$dir/post$x.scale ark:- |" \
        $dir/${type}.$x.acc 2>$dir/acc_${type}.$x.log || exit 1;
    done
    echo "Iteration $x of MCE: computing ml (smoothing) stats"
    gmm-acc-stats $dir/$x.mdl "$feats" "ark:gunzip -c $tmpdir/num.posts.gz |" \
       $dir/ml.$x.acc  2>$dir/acc_ml.$x.log || exit 1;

    echo "Iteration $x of MCE: doing update"
    ( gmm-est-gaussians-ebw $dir/$x.mdl "gmm-ismooth-stats --tau=$tau $dir/ml.$x.acc $dir/num.$x.acc -|" \
           $dir/den.$x.acc - | \
     gmm-est-weights-ebw - $dir/num.$x.acc $dir/den.$x.acc $dir/$[$x+1].mdl ) \
      2>$dir/update.$x.log || exit 1;

    grep Overall $dir/compute_mce_scale.$x.log
  fi
  x=$[$x+1]
done

# Just copy the source-dir's occs, in case we later need them for something...
cp $alidir/final.occs $dir
( cd $dir; ln -s $x.mdl final.mdl )


echo Done

