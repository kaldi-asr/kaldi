#!/bin/bash
# Copyright 2010-2011 Chao Weng  Microsoft Corporation  Arnab Ghoshal

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
# This script does MCE training
# This script trains a model on top of LDA + [something] features, where
# [something] may be MLLT, or ET, or MLLT + SAT.  Any speaker-specific
# transforms are expected to be located in the alignment directory. 
# This script never re-estimates any transforms, it just does model 
# training.  To make this faster, it initializes the model from the
# old system's model, i.e. for each p.d.f., it takes the best-match pdf
# from the old system (based on overlap of tree-stats counts), and 
# uses that GMM to initialize the current GMM.

niters=4
nj=4
tau=100
mce_alpha=0.1 # Constant used in MCE computation
mce_beta=0.0 # ditto; will normally be 0.

cmd=scripts/run.pl
prunepost=0.1 # This controls random pruning of posteriors...
    # this is done mainly to reduce disk usage (temporary storage for
    # posteriors in lattices).  Larger -> less disk usage.
acwt=0.1
scale_opts="--transition-scale=1.0 --self-loop-scale=0.1"
stage=-1

for x in `seq 8`; do
  if [ $1 == "--num-jobs" ]; then
    shift; nj=$1; shift
  fi
  if [ $1 == "--num-iters" ]; then
    shift; niters=$1; shift
  fi
  if [ $1 == "--boost" ]; then
    shift; boost=$1; shift
  fi
  if [ $1 == "--cmd" ]; then
    shift; cmd=$1; shift
    [ -z "$cmd" ] && echo Empty argument to --cmd option && exit 1;
  fi  
  if [ $1 == "--acwt" ]; then
    shift; acwt=$1; shift
  fi  
  if [ $1 == "--stage" ]; then
    shift; stage=$1; shift
  fi  
done

if [ $# != 6 ]; then
   echo "Usage: steps/train_lda_etc_mce.sh <data-dir> <lang-dir> <ali-dir> <denlat-dir> <model-dir> <exp-dir>"
   echo " e.g.: steps/train_lda_etc_mce.sh data/train_si84 data/lang exp/tri2b_ali_si84 exp/tri2b_denlats_si84 exp/tri2b exp/tri2b_mmi"
   exit 1;
fi

if [ -f path.sh ]; then . path.sh; fi

data=$1
lang=$2
alidir=$3
denlatdir=$4
srcdir=$5 # may be same model as in alidir, but may not be, e.g.
      # if you want to test MMI with different #iters.
dir=$6
tmpdir=$dir/posts  # Used to store posteriors on each iteration.
mkdir -p $tmpdir
mkdir -p $dir/scores $dir/scales
silphonelist=`cat $lang/silphones.csl` || exit 1;
oov_sym=`cat $lang/oov.txt` || exit 1;
mkdir -p $dir/log

if [ ! -f $srcdir/final.mdl -o ! -f $srcdir/final.mat ]; then
  echo "Error: alignment dir $alidir does not contain one of final.mdl or final.mat"
  exit 1;
fi
cp $srcdir/final.mat $srcdir/tree $dir

n=`get_splits.pl $nj | awk '{print $1}'`
if [ -f $alidir/$n.trans ]; then
  use_trans=true
  echo Using transforms from directory $alidir
else
  echo No transforms present in alignment directory: assuming speaker independent.
  use_trans=false
fi

for n in `get_splits.pl $nj`; do
  featspart[$n]="ark,s,cs:apply-cmvn --norm-vars=false --utt2spk=ark:$data/split$nj/$n/utt2spk ark:$alidir/$n.cmvn scp:$data/split$nj/$n/feats.scp ark:- | splice-feats ark:- ark:- | transform-feats $alidir/final.mat ark:- ark:- |"
  $use_trans && featspart[$n]="${featspart[$n]} transform-feats --utt2spk=ark:$data/split$nj/$n/utt2spk ark:$alidir/$n.trans ark:- ark:- |"

  [ ! -f $denlatdir/lat.$n.gz ] && echo No such file $denlatdir/lat.$n.gz && exit 1;
done

rm $dir/.error 2>/dev/null

if [ $stage -le -1 ]; then
  echo Creating numerator lattices
  if [ ! -f $dir/LG.fst -o $dir/LG.fst -ot $denlatdir/lang/L.fst ]; then
    fsttablecompose $denlatdir/lang/L.fst $denlatdir/lang/G.fst > $dir/LG.fst || exit 1
  fi
  for n in `get_splits.pl $nj`; do
    tra="ark:scripts/sym2int.pl --map-oov \"$oov_sym\" --ignore-first-field $lang/words.txt < $data/split$nj/$n/text |"
    # Note: use beam=15 not 10 as beam as we don't have retry option here.
    # lattice-beam doesn't matter-- will only be one path.
    $cmd $dir/log/decode_num.$n.log \
      compile-train-graphs $scale_opts $dir/tree $alidir/final.mdl $dir/LG.fst "$tra" ark:- \| \
      gmm-latgen-faster --beam=15 --lattice-beam=1 --acoustic-scale=$acwt \
        --word-symbol-table=$denlatdir/lang/words.txt $alidir/final.mdl ark:- "${featspart[$n]}" \
       "ark:|gzip -c >$dir/numlat.$n.gz" || touch $dir/.error &
  done
  wait
  [ -f $dir/.error ] && echo "Error creating numerator lattices" && exit 1;
fi  


cur_mdl=$srcdir/final.mdl
x=0
while [ $x -lt $niters ]; do
  # Get denominator stats...  For simplicity we rescore the lattice
  # on all iterations, even though it shouldn't be necessary on the zeroth
  # (but we want this script to work even if $srcdir doesn't contain the
  # model used to generate the lattice).
  if [ $stage -le $x ]; then
    echo "Iteration $x: getting denominator posts and scores."
    for n in `get_splits.pl $nj`; do  
      denlats="ark:lattice-difference 'ark:gunzip -c $denlatdir/lat.$n.gz|' 'ark:gunzip -c $dir/numlat.$n.gz|' ark:- |"
      $cmd $dir/log/post_den.$x.$n.log \
        gmm-rescore-lattice $cur_mdl "$denlats" "${featspart[$n]}" ark:- \| \
        lattice-to-post --acoustic-scale=$acwt ark:- ark:- ark,t:$dir/scores/den.$x.$n.scores \| \
        rand-prune-post $prunepost ark:- "ark:|gzip -c > $tmpdir/den_posts.$n.gz" \
          || touch $dir/.error &
    done 
    wait
    [ -f $dir/.error ] && echo Error accumulating den posts and scores on iter $x && exit 1;

    echo "Iteration $x: getting numerator posts and scores."
    for n in `get_splits.pl $nj`; do  
      $cmd $dir/log/post_num.$x.$n.log \
        gmm-rescore-lattice $cur_mdl "ark:gunzip -c $dir/numlat.$n.gz|" "${featspart[$n]}" ark:- \| \
        lattice-to-post --acoustic-scale=$acwt ark:- "ark:|gzip -c > $tmpdir/num_posts.$n.gz" \
          ark,t:$dir/scores/num.$x.$n.scores || touch $dir/.error &
    done 
    wait
    [ -f $dir/.error ] && echo Error accumulating den posts and scores on iter $x && exit 1;

    echo "Iteration $x: getting MCE scaling factors"
    for n in `get_splits.pl $nj`; do  
      compute-mce-scale --mce-alpha=$mce_alpha --mce-beta=$mce_beta ark:$dir/scores/num.$x.$n.scores \
          ark:$dir/scores/den.$x.$n.scores ark:$dir/scales/$x.$n.scales \
      2>$dir/log/compute_mce_scale.$x.$n.log || exit 1;
    done

    # Compute the scales all together and discard the result, to get the MCE criterion in
    # one log file.
    compute-mce-scale --mce-alpha=$mce_alpha --mce-beta=$mce_beta "ark:cat $dir/scores/num.$x.*.scores|" \
        "ark:cat $dir/scores/den.$x.*.scores|" ark:/dev/null  2>$dir/log/mce_crit.$x.log || exit 1;

    grep Overall $dir/log/mce_crit.$x.log

    for type in num den; do
      echo "Iteration $x: getting $type stats"
      for n in `get_splits.pl $nj`; do  
        $cmd $dir/log/acc_$type.$x.$n.log \
          gmm-acc-stats $cur_mdl "${featspart[$n]}" \
           "ark:gunzip -c $tmpdir/${type}_posts.$n.gz|scale-post ark:- ark:$dir/scales/$x.$n.scales ark:-|" \
            $dir/${type}.$x.$n.acc || touch $dir/.error &
      done
      wait
      [ -f $dir/.error ] && echo Error getting $type stats on iter $x && exit 1;
      $cmd $dir/log/${type}_acc_sum.$x.log \
        gmm-sum-accs $dir/${type}.$x.acc $dir/${type}.$x.*.acc || exit 1;
      rm $dir/${type}.$x.*.acc
    done   
    echo "Iteration $x: getting ml (smoothing) stats"
    for n in `get_splits.pl $nj`; do  
      $cmd $dir/log/acc_ml.$x.$n.log \
        gmm-acc-stats $cur_mdl "${featspart[$n]}" \
         "ark:gunzip -c $tmpdir/num_posts.$n.gz|" $dir/ml.$x.$n.acc || touch $dir/.error &
    done
    wait
    [ -f $dir/.error ] && echo Error getting ml stats on iter $x && exit 1;

    $cmd $dir/log/ml_acc_sum.$x.log \
      gmm-sum-accs $dir/ml.$x.acc $dir/ml.$x.*.acc || exit 1;
    rm $dir/ml.$x.*.acc

    $cmd $dir/log/update.$x.log \
      gmm-est-gaussians-ebw $cur_mdl "gmm-ismooth-stats --tau=$tau $dir/ml.$x.acc $dir/num.$x.acc -|" \
        $dir/den.$x.acc - \| \
      gmm-est-weights-ebw - $dir/num.$x.acc $dir/den.$x.acc $dir/$[$x+1].mdl || exit 1;
    rm $dir/{ml,num,den}.$x.acc
  else 
    echo "Not doing this iteration because --stage=$stage"
  fi
  cur_mdl=$dir/$[$x+1].mdl

  x=$[$x+1]
done

echo "Succeeded with $niters iterations of MCE training"

( cd $dir; ln -s $x.mdl final.mdl )
exit 0;
