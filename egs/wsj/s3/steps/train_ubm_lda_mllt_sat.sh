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


# Warning: this is fundamentally different from train_ubm_lda_etc.sh
# (more than the name would suggest).

# This UBM training recipe starts from an LDA+MLLT system (or possibly
# LDA+MLLT+SAT; it doesn't use the transforms though, or the alignment model.
# This script is different from 
# train_ubm_lda_etc.sh, and it produces more outputs; it's intended to be
# used for where the SAT associated with the SGMMs is done at the GMM level,
# without a separate decoding pass (except for speech-silence detection, which
# we do at the frame level).

# We first obtain a diagonal mixture-of-Gaussians by clustering a conventional
# LDA+MLLT system.
# We then do 4 iterations of diagonal-GMM training, with fMLLR transform
# re-estimation on each iteration.  [Note that transform estimation done only
# on non-silence, based on the supplied alignments].

# At the end we do two iterations of full-covariance GMM training (without allowing
# any Gaussians to be deleted in the update phase, so we retain the correspondence
# of the Gaussians with the diagonal ones.

# Starting from the final diagonal GMM, we also do a single iteration of 
# weight re-estimation (keeping the structure the same) on silence and non-silence
# separately.  This will enable us to do speech/silence detection in order
# to estimate the transforms.

# Train UBM from a trained HMM/GMM system [with splice+LDA+[MLLT/ET/MLLT+SAT] features]
# Alignment directory is used for the CMN and transforms.
# A UBM is just a single mixture of Gaussians (full-covariance, in our case), that's trained
# on all the data.  This will later be used in Subspace Gaussian Mixture Model (SGMM)
# training.

nj=4
cmd=scripts/run.pl
realsilphonelist=
stage=-2

for x in `seq 3`; do
  if [ $1 == "--num-jobs" ]; then
     shift
     nj=$1
     shift
  fi
  if [ $1 == "--realsilphonelist" ]; then # colon-separated list of 
     shift  # numeric id's of silence phones (but not vocalized noise, laughter...)
     realsilphonelist=$1
     shift
  fi
  if [ $1 == "--cmd" ]; then
     shift
     cmd=$1
     shift
  fi  
  if [ $1 == "--stage" ]; then
    stage=$2
    shift; shift
  fi
done

if [ $# != 5 ]; then
  echo "Usage: steps/train_ubm_lda_sat.sh <num-comps> <data-dir> <lang-dir> <ali-dir> <exp-dir>"
  echo " e.g.: steps/train_ubm_lda_sat.sh 400 data/train_si84 data/lang exp/tri2b_ali_si84 exp/ubm3c"
  exit 1;
fi

if [ -f path.sh ]; then . path.sh; fi

numcomps=$1
data=$2
lang=$3
alidir=$4
dir=$5

ngselect=25

silphonelist=`cat $lang/silphones.csl`
[ -z "$realsilphonelist" ] && realsilphonelist=$silphonelist # set to regular  silence-phone list,
# if not set on command line to just silence.

mkdir -p $dir/log
cp $alidir/final.mat $dir

if [ ! -d $data/split$nj -o $data/split$nj -ot $data/feats.scp ]; then
  scripts/split_data.sh $data $nj
fi

for n in `get_splits.pl $nj`; do
  sifeatspart[$n]="ark,s,cs:apply-cmvn --norm-vars=false --utt2spk=ark:$data/split$nj/$n/utt2spk ark:$alidir/$n.cmvn scp:$data/split$nj/$n/feats.scp ark:- | splice-feats ark:- ark:- | transform-feats $alidir/final.mat ark:- ark:- |"
  featspart[$n]="${sifeatspart[$n]}"
done

intermediate=2000
if [ $[$numcomps*2] -gt $intermediate ]; then
  intermediate=$[$numcomps*2];
fi


if [ $stage -le -2 ]; then
  echo "Clustering model $alidir/final.mdl to get initial (diagonal) UBM"
  # typically: --intermediate-numcomps=2000 --ubm-numcomps=400

  $cmd $dir/log/cluster.log \
    init-ubm --intermediate-numcomps=$intermediate --ubm-numcomps=$numcomps \
     --verbose=2 --fullcov-ubm=false $alidir/final.mdl $alidir/final.occs \
      $dir/0.dubm   || exit 1;
fi

rm $dir/.error 2>/dev/null

if [ $stage -le -1 ]; then
  # First do Gaussian selection to 25 components, which will be used
  # as the initial screen for all further passes.
  for n in `get_splits.pl $nj`; do
    $cmd $dir/log/gselect.$n.log \
      gmm-gselect --n=$ngselect $dir/0.dubm "${featspart[$n]}" \
        "ark:|gzip -c >$dir/gselect.$n.gz"  &
  done
  wait
  [ -f $dir/.error ] && echo "Error doing GMM selection" && exit 1 
fi

for x in 0 1 2 3; do
  if [ $stage -le $x ]; then
    # First estimate transforms
    echo "Pass $x: estimating transforms"
    for n in `get_splits.pl $nj`; do
      $cmd $dir/log/fmllr.$x.$n.log \
        gmm-global-est-fmllr --spk2utt=ark:$data/split$nj/$n/spk2utt \
          "--gselect=ark,s,cs:gunzip -c $dir/gselect.$n.gz|" \
          "--weights=ark,s,cs:gunzip -c $alidir/$n.ali.gz | ali-to-post ark:- ark:- | weight-silence-post 0.01 $silphonelist $alidir/final.mdl ark:- ark:- | post-to-weights ark:- ark:- |" \
          $dir/$x.dubm "${featspart[$n]}" "ark:$dir/$n.trans.tmp" || touch $dir/.error &
    done
    wait;
    [ -f $dir/.error ] && echo "Error doing fMLLR computation" && exit 1;   
    rm $dir/log/compose.$x.log 2>/dev/null
    for n in `get_splits.pl $nj`; do 
      featspart[$n]="${sifeatspart[$n]} transform-feats --utt2spk=ark:$data/split$nj/$n/utt2spk ark:$dir/$n.trans ark:- ark:- |"
      if [ $x == 0 ]; then
        mv $dir/$n.trans.tmp $dir/$n.trans || exit 1;
      else
        compose-transforms --b-is-affine=true ark:$dir/$n.trans.tmp ark:$dir/$n.trans ark:$dir/$n.trans.tmp2 2>>$dir/log/compose.$x.log || exit 1;
        mv $dir/$n.trans.tmp2 $dir/$n.trans;
        rm $dir/$n.trans.tmp
      fi
    done

    echo "Pass $x: doing GMM accumulation"
    for n in `get_splits.pl $nj`; do
      $cmd $dir/log/acc.$x.$n.log \
        gmm-global-acc-stats  "--gselect=ark,s,cs:gunzip -c $dir/gselect.$n.gz|" \
           $dir/$x.dubm "${featspart[$n]}" $dir/$x.$n.acc || touch $dir/.error &
    done
    wait
    [ -f $dir/.error ] && echo "Error accumulating stats for diagonal UBM estimation on pass $x" && exit 1;
    $cmd $dir/log/update.$x.log \
      gmm-global-est --remove-low-count-gaussians=false --verbose=2 $dir/$x.dubm "gmm-global-sum-accs - $dir/$x.*.acc |" \
        $dir/$[$x+1].dubm || exit 1;
    rm $dir/$x.*.acc # $dir/$x.dubm
  fi
done

# This is only necessary if you did the --stage stuff.
for n in `get_splits.pl $nj`; do 
  featspart[$n]="${sifeatspart[$n]} transform-feats --utt2spk=ark:$data/split$nj/$n/utt2spk ark:$dir/$n.trans ark:- ark:- |"
done


gmm-global-to-fgmm $dir/4.dubm $dir/4.ubm 2>$dir/log/convert.4.log || exit 1;

# Now we do two iterations of full-covariance accumulation and update
# (with fixed transforms)-- being careful to not let any Gaussians be removed,
# as we need to maintain a one-to-one correspondence.

for x in 4 5; do
  if [ $stage -le $x ]; then
    echo "Doing full-covariance accumulation (iter $x)"
    for n in `get_splits.pl $nj`; do
      $cmd $dir/log/acc.$x.$n.log \
        fgmm-global-acc-stats "--gselect=ark,s,cs:gunzip -c $dir/gselect.$n.gz|" \
           $dir/$x.ubm "${featspart[$n]}" $dir/$x.$n.acc || touch $dir/.error &
    done
    wait
    [ -f $dir/.error ] && echo "Error accumulating stats for UBM estimation on pass $x" && exit 1;
    $cmd $dir/log/update.$x.log \
      fgmm-global-est --remove-low-count-gaussians=false \
         --verbose=2 $dir/$x.ubm "fgmm-global-sum-accs - $dir/$x.*.acc |" \
        $dir/$[$x+1].ubm || exit 1;
    rm $dir/$x.*.acc $dir/$x.ubm
  fi
done


# We'll make 6.dubm which is the diagonal version of it, and
# 6.si.{ubm,dubm} which is the same in the speaker-independent space
# also make 6.si.s.ubm and 6.si.ns.ubm (non-silence and silence-weighted UBM's,
# also in the speaker-independent space).

fgmm-global-to-gmm $dir/6.ubm $dir/6.dubm 2>$dir/log/convert.6.log || exit 1;

if [ $stage -le 6 ]; then 
  # Compute full and diagonal UBMs in speaker-independent space, using
  # Gaussian alignments from full UBM.  First compute full UBM in SI space,
  # then make diagonal one.
  x=6
  echo "Getting SI stats to make SI UBM."
  for n in `get_splits.pl $nj`; do
    $cmd $dir/log/acc.$x.$n.log \
      fgmm-global-acc-stats-twofeats "--gselect=ark,s,cs:gunzip -c $dir/gselect.$n.gz|" \
        $dir/$x.ubm "${featspart[$n]}" "${sifeatspart[$n]}" $dir/$x.si.$n.acc || touch $dir/.error &          
  done
  wait
  [ -f $dir/.error ] && echo "Error accumulating SI stats for final UBM estimation" && exit 1;
  $cmd $dir/log/update.$x.si.log \
    fgmm-global-est --remove-low-count-gaussians=false \
       --verbose=2 $dir/$x.ubm "fgmm-global-sum-accs - $dir/$x.si.*.acc |" \
      $dir/$x.si.ubm || exit 1;
  rm $dir/$x.si.*.acc
fi

fgmm-global-to-gmm $dir/6.si.ubm $dir/6.si.dubm 2>$dir/log/convert.6.si.log || exit 1;

if [ $stage -le 7 ]; then
  x=6
  flags=mw
  # When estimating the silence and non-silence models, we only estimate
  # the means and weights (as is customary in speaker-id); the covariances are
  # left at their values in the generic speaker-independent model.
  for type in si.s si.ns; do
    echo "Getting SI stats to make UBM of type $type."
    if [ $type == "si.s" ]; then reverse=true; silp=$realsilphonelist; else reverse=false; silp=$silphonelist; fi
    for n in `get_splits.pl $nj`; do
      $cmd $dir/log/acc.$x.$type.$n.log \
        fgmm-global-acc-stats-twofeats --update-flags=$flags \
         "--gselect=ark,s,cs:gunzip -c $dir/gselect.$n.gz|" \
         "--weights=ark,s,cs:gunzip -c $alidir/$n.ali.gz | ali-to-post ark:- ark:- | weight-silence-post 0.0 $silp $alidir/final.mdl ark:- ark:- | post-to-weights ark:- ark:- | reverse-weights --reverse=$reverse ark:- ark:- |" \
         $dir/$x.ubm "${featspart[$n]}" "${sifeatspart[$n]}" $dir/$x.$type.$n.acc || touch $dir/.error &
    done
    wait
    [ -f $dir/.error ] && echo "Error accumulating stats for final UBM estimation, type $type" && exit 1;
    $cmd $dir/log/update.$x.$type.log \
      fgmm-global-est --update-flags=$flags --remove-low-count-gaussians=false \
         --verbose=2 $dir/$x.si.ubm "fgmm-global-sum-accs - $dir/$x.$type.*.acc |" \
        $dir/$x.$type.ubm || exit 1;
    rm $dir/$x.$type.*.acc
  done
fi

# estimate silence and non-silence UBMs in the speaker adapted space.
if [ $stage -le 8 ]; then
  x=6
  flags=mw
  # When estimating the silence and non-silence models, we only estimate
  # the means and weights (as is customary in speaker-id); the covariances are
  # left at their values in the generic speaker-independent model.
  for type in s ns; do
    echo "Getting stats to make UBM of type $type."
    if [ $type == "s" ]; then reverse=true; silp=$realsilphonelist; else reverse=false; silp=$silphonelist; fi
    for n in `get_splits.pl $nj`; do
      $cmd $dir/log/acc.$x.$type.$n.log \
        fgmm-global-acc-stats --update-flags=$flags \
         "--gselect=ark,s,cs:gunzip -c $dir/gselect.$n.gz|" \
         "--weights=ark,s,cs:gunzip -c $alidir/$n.ali.gz | ali-to-post ark:- ark:- | weight-silence-post 0.0 $silp $alidir/final.mdl ark:- ark:- | post-to-weights ark:- ark:- | reverse-weights --reverse=$reverse ark:- ark:- |" \
         $dir/$x.ubm "${featspart[$n]}" $dir/$x.$type.$n.acc || touch $dir/.error &
    done
    wait
    [ -f $dir/.error ] && echo "Error accumulating stats for final UBM estimation, type $type" && exit 1;
    $cmd $dir/log/update.$x.$type.log \
      fgmm-global-est --update-flags=$flags --remove-low-count-gaussians=false \
         --verbose=2 $dir/$x.ubm "fgmm-global-sum-accs - $dir/$x.$type.*.acc |" \
        $dir/$x.$type.ubm || exit 1;
    rm $dir/$x.$type.*.acc
  done
fi


for suffix in ubm dubm si.ubm si.dubm si.s.ubm si.ns.ubm s.ubm ns.ubm; do
  rm $dir/final.$suffix 2>/dev/null
  ln -s 6.$suffix $dir/final.$suffix
done


rm $dir/gselect.*.gz



