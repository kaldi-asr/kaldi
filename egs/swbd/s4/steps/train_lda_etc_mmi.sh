#!/bin/bash
# Copyright 2010-2011 Microsoft Corporation  Arnab Ghoshal

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
# This script does MMI training
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
boost=0.0
cmd=scripts/run.pl
acwt=0.1

for x in 1 2 3; do
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
done

if [ $# != 6 ]; then
   echo "Usage: steps/train_lda_etc_mmi.sh <data-dir> <lang-dir> <ali-dir> <denlat-dir> <model-dir> <exp-dir>"
   echo " e.g.: steps/train_lda_etc_mmi.sh data/train_si84 data/lang exp/tri2b_ali_si84 exp/tri2b_denlats_si84 exp/tri2b exp/tri2b_mmi"
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
  echo "Iteration $x: getting denominator stats."
  # Get denominator stats...  For simplicity we rescore the lattice
  # on all iterations, even though it shouldn't be necessary on the zeroth
  # (but we want this script to work even if $srcdir doesn't contain the
  # model used to generate the lattice).
  for n in `get_splits.pl $nj`; do  
    $cmd $dir/log/acc_den.$x.$n.log \
      gmm-rescore-lattice $cur_mdl "${latspart[$n]}" "${featspart[$n]}" ark:- \| \
      lattice-to-post --acoustic-scale=$acwt ark:- ark:- \| \
      gmm-acc-stats $cur_mdl "${featspart[$n]}" ark:- $dir/den_acc.$x.$n.acc \
       || touch $dir/.error &
  done 
  wait
  [ -f $dir/.error ] && echo Error accumulating den stats on iter $x && exit 1;
  $cmd $dir/log/den_acc_sum.$x.log \
    gmm-sum-accs $dir/den_acc.$x.acc $dir/den_acc.$x.*.acc || exit 1;
  rm $dir/den_acc.$x.*.acc

  echo "Iteration $x: getting numerator stats."
  for n in `get_splits.pl $nj`; do  
    $cmd $dir/log/acc_num.$x.$n.log \
      gmm-acc-stats-ali $cur_mdl "${featspart[$n]}" "ark:gunzip -c $alidir/$n.ali.gz|" \
        $dir/num_acc.$x.$n.acc || touch $dir/.error &
  done
  wait;
  [ -f $dir/.error ] && echo Error accumulating num stats on iter $x && exit 1;
  $cmd $dir/log/num_acc_sum.$x.log \
    gmm-sum-accs $dir/num_acc.$x.acc $dir/num_acc.$x.*.acc || exit 1;
  rm $dir/num_acc.$x.*.acc

  $cmd $dir/log/update.$x.log \
    gmm-est-mmi $cur_mdl $dir/num_acc.$x.acc $dir/den_acc.$x.acc $dir/$[$x+1].mdl \
    || exit 1;

  cur_mdl=$dir/$[$x+1].mdl

  # Some diagnostics
  den=`grep Overall $dir/log/acc_den.$x.*.log  | grep lattice-to-post | awk '{p+=$7*$9; nf+=$9;} END{print p/nf;}'`
  num=`grep Overall $dir/log/acc_num.$x.*.log  | grep gmm-acc-stats-ali | awk '{p+=$11*$13; nf+=$13;} END{print p/nf}'`
  diff=`perl -e "print ($num * $acwt - $den);"`
  impr=`grep Overall $dir/log/update.$x.log | awk '{print $10;}'`
  impr=`perl -e "print ($impr * $acwt);"` # auxf impr normalized by multiplying by
  # kappa, so it's comparable to an objective-function change.
  echo On iter $x, objf was $diff, auxf improvement was $impr | tee $dir/objf.$x.log

  x=$[$x+1]
done

echo "Succeeded with $niters iterations of MMI training (boosting factor = $boost)"

( cd $dir; ln -s $x.mdl final.mdl )
