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
# This does mixing-up, three iterations of model training, realignment,
# and two more iterations of model training.
# It's intended to be used for experiments with LDA+MLLT or LDA+MLLT+SAT
# models where you increase the number of mixtures and see if it helps.

nj=4
cmd=scripts/run.pl
for x in 1 2; do
  if [ "$1" == "--num-jobs" ]; then
    shift
    nj=$1
    shift
  fi
  if [ "$1" == "--cmd" ]; then
    shift
    cmd=$1
    shift
  fi  
done

if [ $# != 5 ]; then
   echo "Usage: steps/mixup_lda_etc.sh <num-gauss> <data-dir> <old-exp-dir> <alignment-dir> <exp-dir>"
   echo "Note: <alignment-dir> is only provided so we can get the CMVN data from there."
   echo " e.g.: steps/mixup_lda_etc.sh 20000 data/train_si84 exp/tri3b exp/tri2b_ali_si84 exp/tri3b_20k"
   exit 1;
fi

if [ -f path.sh ]; then . path.sh; fi

numgauss=$1
data=$2
olddir=$3
alidir=$4 # only needed for CMVN data.
dir=$5

for f in $data/feats.scp $olddir/final.mdl $olddir/final.mat; do
  [ ! -f $f ] && echo "mixup_lda_etc.sh: no such file $f" && exit 1;
done

scale_opts="--transition-scale=1.0 --acoustic-scale=0.1 --self-loop-scale=0.1"

mkdir -p $dir/log
cp $olddir/final.mat $olddir/tree $dir/

if [ ! -d $data/split$nj -o $data/split$nj -ot $data/feats.scp ]; then
  echo "Splitting data-dir $data into $nj pieces, but watch out: we require #jobs"  \
      "to be matched with $olddir"
  split_data.sh $data $nj
fi

for n in `get_splits.pl $nj`; do
  sifeatspart[$n]="ark,s,cs:apply-cmvn --norm-vars=false --utt2spk=ark:$data/split$nj/$n/utt2spk ark:$alidir/$n.cmvn scp:$data/split$nj/$n/feats.scp ark:- | splice-feats ark:- ark:- | transform-feats $dir/final.mat ark:- ark:- |"
  featspart[$n]="${sifeatspart[$n]}"
done

ln.pl $olddir/*.fsts.gz $dir  # Link FSTs.

# Adjust the features to reflect any transforms we may have in $olddir.
first=`get_splits.pl $nj | awk '{print $1}'`
if [ -f $olddir/$first.trans ]; then
  ln.pl $olddir/*.trans $dir # Link transforms, in case we do something that requires them.
  # This program "ln.pl" uses relative links, in case we move the whole directory tree.
  have_trans=true
  echo "Using transforms in $olddir (linking to $dir)"
  for n in `get_splits.pl $nj`; do
    featspart[$n]="${sifeatspart[$n]} transform-feats --utt2spk=ark:$data/split$nj/$n/utt2spk ark:$dir/$n.trans ark:- ark:- |"
  done
else
  have_trans=false
  echo "No transforms in $olddir, assuming you are not using fMLLR."
fi


echo Mixing up old model to $numgauss Gaussians
$cmd $dir/log/mixup.log \
  gmm-mixup --mix-up=$numgauss $olddir/final.mdl $olddir/final.occs $dir/0.mdl || exit 1;

rm $dir/.error 2>/dev/null

dir_for_alignments=$olddir # This is where we find the alignments...
      # after we realign, on iter 3, we'll use the ones in $dir
niters=4
for x in `seq 0 $niters`; do  # Do five iterations of E-M; on 3rd iter, realign.
  echo Iteration $x
  if [ $x -eq 2 ]; then
    echo Realigning data on iteration $x
    for n in `get_splits.pl $nj`; do
      [ ! -f $dir/$n.fsts.gz ] && echo Expecting FSTs to exist: no such file $dir/$n.fsts.gz \
        && exit 1;
      $cmd $dir/log/align.$x.$n.log \
        gmm-align-compiled $scale_opts --beam=10 --retry-beam=40 $dir/$x.mdl \
          "ark:gunzip -c $dir/$n.fsts.gz|" "${featspart[$n]}" \
          "ark:|gzip -c >$dir/$n.ali.gz" || touch $dir/.error &
    done
    wait
    [ -f $dir/.error ] && echo "Error computing alignments" && exit 1;
    dir_for_alignments=$dir
  fi
  echo "Accumulating statistics"
  for n in `get_splits.pl $nj`; do  
     $cmd $dir/log/acc.$x.$n.log \
      gmm-acc-stats-ali  $dir/$x.mdl "${featspart[$n]}" \
        "ark,s,cs:gunzip -c $dir_for_alignments/$n.ali.gz|" $dir/$x.$n.acc || touch $dir/.error &
  done
  wait;
  [ -f $dir/.error ] && echo "Error accumulating stats on iteration $x" && exit 1;
  $cmd $dir/log/update.$x.log \
    gmm-est --write-occs=$dir/$[$x+1].occs $dir/$x.mdl \
      "gmm-sum-accs - $dir/$x.*.acc |" $dir/$[$x+1].mdl || exit 1;
  rm $dir/$x.mdl $dir/$x.*.acc
  rm $dir/$x.occs  2>/dev/null
done
x=$[$niters+1]
rm $dir/final.mdl $dir/final.occs 2>/dev/null
ln -s $x.mdl $dir/final.mdl
ln -s $x.occs $dir/final.occs

if $have_trans; then # we have transforms, so compute the alignment model,
# which is as the model but with
# the default features (shares Gaussian-level alignments).
  for n in `get_splits.pl $nj`; do
    $cmd $dir/log/acc_alimdl.$n.log \
    ali-to-post "ark:gunzip -c $dir_for_alignments/$n.ali.gz|" ark:-  \| \
      gmm-acc-stats-twofeats $dir/$x.mdl "${featspart[$n]}" "${sifeatspart[$n]}" \
        ark,s,cs:- $dir/$x.$n.acc2 || touch $dir/.error &
  done
  wait;
  [ -f $dir/.error ] && echo "Error accumulating alignment statistics." && exit 1;
  # Update model.
  $cmd $dir/log/est_alimdl.log \
    gmm-est --write-occs=$dir/final.occs --remove-low-count-gaussians=false $dir/$x.mdl \
    "gmm-sum-accs - $dir/$x.*.acc2|" $dir/$x.alimdl  || exit 1;
  rm $dir/$x.*.acc2

  rm $dir/final.alimdl 2>/dev/null
  ln -s $x.alimdl $dir/final.alimdl 
fi


# Print out summary of the warning messages.
for x in $dir/log/*.log; do 
  n=`grep WARNING $x | wc -l`; 
  if [ $n -ne 0 ]; then echo $n warnings in $x; fi; 
done

echo Done
