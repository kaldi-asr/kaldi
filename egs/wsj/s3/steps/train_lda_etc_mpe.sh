#!/bin/bash
# Copyright 2010-2011 Microsoft Corporation  Arnab Ghoshal  Chao Weng

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#  http://www.apache.org/licenses/LICENSE-2.0
#
# THIS CODE IS PROVIDED *AS IS* BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, EITHER EXPRESS OR IMPLIED, INCLUDING WITHOUT LIMITATION ANY IMPLIED # WARRANTIES OR CONDITIONS OF TITLE, FITNESS FOR A PARTICULAR PURPOSE, # MERCHANTABLITY OR NON-INFRINGEMENT.
# See the Apache 2 License for the specific language governing permissions and
# limitations under the License.

# To be run from ..
# This script does frame level MPE training
# This script trains a model on top of LDA + [something] features, where # [something] may be MLLT, or ET, or MLLT + SAT.  Any speaker-specific
# transforms are expected to be located in the alignment directory. 
# This script never re-estimates any transforms, it just does model 
# training.  

# Basically we are doing 4 iterations of Extended Baum-Welch (EBW)
# estimation, as described in Dan Povey's thesis, with a few differences:
#  (i) The lattices have fixed state-level
# alignments, so there is no forward-backward going on within phones (the only
# forward-backward is a lattice-level forward-backward with fixed Viterbi
# alignments).
# (ii) There are no numerator lattices, the input for numerator is just posteriors
# which can be generated from fixed alignments or lattices (we just keep this option
# open). 


niters=4
nj=4
boost=0.0
tau=200
merge=true # if true, cancel num and den counts as described in 
cmd=scripts/run.pl
acwt=0.1
stage=0
scale_opts="--transition-scale=1.0 --acoustic-scale=0.1 --self-loop-scale=0.1"

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
  if [ $1 == "--tau" ]; then
    shift; tau=$1; shift
  fi  
  if [ $1 == "--stage" ]; then
    shift; stage=$1; shift
  fi  
done

if [ $# != 6 ]; then
   echo "Usage: steps/train_lda_etc_mpe.sh <data-dir> <lang-dir> <ali-dir> <denlat-dir> <model-dir> <exp-dir>"
   echo " e.g.: steps/train_lda_etc_mpe.sh data/train_si84 data/lang exp/tri2b_ali_si84 exp/tri2b_denlats_si84 exp/tri2b exp/tri2b_mpe"
   exit 1;
fi

if [ -f path.sh ]; then . path.sh; fi

data=$1
lang=$2
alidir=$3
denlatdir=$4
srcdir=$5 # may be same model as in alidir, but may not be, e.g.
      # if you want to test MPE with different #iters.
dir=$6
silphonelist=`cat $lang/silphones.csl`
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
  latspart[$n]="ark:gunzip -c $denlatdir/lat.$n.gz|"
  # note: in next line, doesn't matter which model we use, it's only used to map to phones.
  [ $boost != "0.0" -a $boost != "0" ] && latspart[$n]="${latspart[$n]} lattice-boost-ali --b=$boost --silence-phones=$silphonelist $alidir/final.mdl ark:- 'ark,s,cs:gunzip -c $alidir/$n.ali.gz|' ark:- |"
done

 

rm $dir/.error 2>/dev/null
cur_mdl=$srcdir/final.mdl
x=0
while [ $x -lt $niters ]; do
  echo "Iteration $x: getting stats."
  # Get denominator and numerator stats together...    This involves
  # merging the num and den posteriors, and (if $merge==true), canceling
  # the +ve and -ve occupancies on each frame. 
  # For simplicity we rescore the lattice
  # on all iterations, even though it shouldn't be necessary on the zeroth
  # (but we want this script to work even if $srcdir doesn't contain the
  #  model used to generate the lattice).
  if [ $stage -le $x ]; then
    for n in `get_splits.pl $nj`; do 
      $cmd $dir/log/acc.$x.$n.log \
        gmm-rescore-lattice $cur_mdl "${latspart[$n]}" "${featspart[$n]}" ark:- \| \
        lattice-to-mpe-post --acoustic-scale=$acwt $cur_mdl \
          "ark,s,cs:gunzip -c $alidir/$n.ali.gz | ali-to-post ark:- ark:- |" ark:- ark:- \| \
        gmm-acc-stats2 $cur_mdl "${featspart[$n]}" ark,s,cs:- \
          $dir/num_acc.$x.$n.acc $dir/den_acc.$x.$n.acc  || touch $dir/.error &
    done 
    wait
    [ -f $dir/.error ] && echo Error accumulating stats on iter $x && exit 1;
    $cmd $dir/log/den_acc_sum.$x.log \
      gmm-sum-accs $dir/den_acc.$x.acc $dir/den_acc.$x.*.acc || exit 1;
    rm $dir/den_acc.$x.*.acc
    $cmd $dir/log/num_acc_sum.$x.log \
      gmm-sum-accs $dir/num_acc.$x.acc $dir/num_acc.$x.*.acc || exit 1;
    rm $dir/num_acc.$x.*.acc

    # note: this tau value is for smoothing to model parameters;
    # you need to use gmm-ismooth-stats to smooth to the ML stats,
    # but anyway this script does canceling of num and den stats on
    # each frame (as suggested in the Boosted MMI paper) which would
    # make smoothing to ML impossible without accumulating extra stats.
    
    # collect ML stats for I-Smoothing
    for n in `get_splits.pl $nj`; do  
      $cmd $dir/log/acc_ml.$x.$n.log \
        gmm-acc-stats $cur_mdl "${featspart[$n]}" \
          "ark,s,cs:gunzip -c $alidir/$n.ali.gz | ali-to-post ark:- ark:- |" \
          $dir/ml.$x.$n.acc || touch $dir/.error &
    done
    wait
    [ -f $dir/.error ] && echo Error getting ml stats on iter $x && exit 1;
    
    $cmd $dir/log/ml_acc_sum.$x.log \
      gmm-sum-accs $dir/ml.$x.acc $dir/ml.$x.*.acc || exit 1;
    rm $dir/ml.$x.*.acc    
    # M-step, re-estimate the parameters
    $cmd $dir/log/update.$x.log \
      gmm-est-gaussians-ebw $cur_mdl "gmm-ismooth-stats --tau=$tau $dir/ml.$x.acc $dir/num_acc.$x.acc -|" \
        $dir/den_acc.$x.acc - \| \
      gmm-est-weights-ebw - $dir/num_acc.$x.acc $dir/den_acc.$x.acc $dir/$[$x+1].mdl || exit 1;
  else 
    echo "not doing this iteration because --stage=$stage"
  fi
  cur_mdl=$dir/$[$x+1].mdl

  # Some diagnostics.. note, this objf is somewhat comparable to the
  # MMI objective function divided by the acoustic weight, and differences in it
  # are comparable to the auxf improvement printed by the update program.
  objf=`grep Overall $dir/log/acc.$x.*.log | grep gmm-acc-stats2 | awk '{ p+=$10*$12; nf+=$12; } END{print p/nf;}'`
  nf=`grep Overall $dir/log/acc.$x.*.log | grep gmm-acc-stats2 | awk '{ nf+=$12; } END{print nf;}'`
  impr=`grep Overall $dir/log/update.$x.log | head -1 | awk '{print $10*$12;}'`
  impr=`perl -e "print ($impr/$nf);"` # renormalize by "real" #frames, to correct
    # for the canceling of stats.
  echo On iter $x, objf was $objf, auxf improvement from MMI was $impr | tee $dir/objf.$x.log

  x=$[$x+1]
done

echo "Succeeded with $niters iterations of MMI training (boosting factor = $boost)"

( cd $dir; rm final.mdl; ln -s $x.mdl final.mdl )
exit 0;
