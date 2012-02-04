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

# This script computes fMLLR transforms, not from a HMM-GMM but from 
# a simple GMM.  It takes a UBM directory as produced by 
# train_ubm_lda_mllt_sat.sh, which contains the files:
# final.{ubm,dubm,si.ubm,si.dubm,si.s.ubm,si.ns.ubm},
# where "ubm" means full-covariance GMMs, "dubm" means
# diagonal-covariance GMM, "si" means speaker-independent
# (i.e. GMMs from the speaker-independent space), "s" means
# trained on silence only, and "ns" means trained on non-silence
# only.
# the general process is:
# Use final.si.dubm to get Gaussian selection info (used for
# all other phases).
# Use final.si.{s,ns}.dubm to get scores for each frame, for
# the silence and non-silence models.
# Process this into silence-weights (probabilities) for each 
# frame.
# Use final.si.ubm and final.ubm in a "two-model" fMLLR estimation,
# weighted by the probability of non-silence on each frame.

# This script does training-data alignment given a model built using 
# [e.g. MFCC] + CMN + LDA + MLLT + SAT features.  It splits the data into
# four chunks and does everything in parallel on the same machine.
# Its output, all in its own experimental directory, is (assuming
# you don't change the #jobs with --num-job option),
# {0,1,2,3}.cmvn {0,1,2,3}.ali.gz, {0,1,2,3}.trans, tree, final.mdl ,
# final.mat and final.occs (the last four are just copied from the source directory). 


nj=4
stage=0
cmd=scripts/run.pl
oldgraphs=false
alidir=
for x in `seq 4`; do
  if [ $1 == "--num-jobs" ]; then
     shift
     nj=$1
     shift
  fi
  if [ $1 == "--stage" ]; then
     shift
     stage=$1
     shift
  fi
  if [ $1 == "--cmd" ]; then
     shift
     cmd=$1
     [ "$cmd" == "" ] && echo "Empty string given to --cmd option" && exit 1;
     shift
  fi  
  if [ $1 == "--fmllr-update-type" ]; then
     shift
     fmllr_update_type=$1 # full|diag|offset|none
     shift
  fi
  if [ $1 == "--check-silence" ]; then
     alidir=$2 # This is used to check the accuracy of the silence
     lang=$3 # detector. [Note: if you're using the same data you used
             # to train the silence model, it's not entirely valid.]
     shift 3;
  fi
done

if [ $# != 3 ]; then
   echo "Usage: steps/get_transforms_from_ubm.sh [options] <data-dir> <ubm-dir> <exp-dir>"
   echo " e.g.: steps/get_transforms_from_ubm.sh data/train data/train exp/ubm3b exp/ubm3b_trans"
   exit 1;
fi

if [ -f path.sh ]; then . path.sh; fi

data=$1
ubmdir=$2
dir=$3

ngselect=25

mkdir -p $dir/log
cp $ubmdir/final.mat $dir/


if [ ! -d $data/split$nj -o $data/split$nj -ot $data/feats.scp ]; then
  scripts/split_data.sh $data $nj
fi

if [ $stage -le 0 ]; then
  echo "Computing cepstral mean and variance statistics"
  for n in `get_splits.pl $nj`; do
    compute-cmvn-stats --spk2utt=ark:$data/split$nj/$n/spk2utt scp:$data/split$nj/$n/feats.scp \
        ark:$dir/$n.cmvn 2>$dir/log/cmvn$n.log || exit 1;
  done
fi

for n in `get_splits.pl $nj`; do
  sifeatspart[$n]="ark,s,cs:apply-cmvn --norm-vars=false --utt2spk=ark:$data/split$nj/$n/utt2spk ark:$dir/$n.cmvn scp:$data/split$nj/$n/feats.scp ark:- | splice-feats ark:- ark:- | transform-feats $dir/final.mat ark:- ark:- |"
done

rm $dir/.error 2>/dev/null
if [ $stage -le 1 ]; then
  echo Getting Gaussian-selection info
  # First do Gaussian selection to 25 components, which will be used
  # as the initial screen for all further passes.
  for n in `get_splits.pl $nj`; do
    $cmd $dir/log/gselect.$n.log \
      gmm-gselect --n=$ngselect $ubmdir/final.si.dubm "${sifeatspart[$n]}" \
        "ark:|gzip -c >$dir/gselect.$n.gz" || touch $dir/.error &
  done
  wait
  [ -f $dir/.error ] && echo "Error doing GMM selection" && exit 1 
fi


# Compute silence probs.
if [ $stage -le 2 ]; then
  echo Getting silence probs
  for n in `get_splits.pl $nj`; do
    $cmd $dir/log/silence_probs.$n.log \
      get-silence-probs \
        --quantize=0.1 \
        "ark,s,cs:fgmm-global-get-frame-likes \"--gselect=ark,s,cs:gunzip -c $dir/gselect.$n.gz|\" $ubmdir/final.si.s.ubm \"${sifeatspart[$n]}\" ark:- |" \
        "ark,s,cs:fgmm-global-get-frame-likes \"--gselect=ark,s,cs:gunzip -c $dir/gselect.$n.gz|\" $ubmdir/final.si.ns.ubm \"${sifeatspart[$n]}\" ark:- |" \
        "ark,t:|gzip -c >$dir/silprobs.$n.gz" || touch $dir/.error &
  done
  wait
  [ -f $dir/.error ] && echo "Error getting silence probs" && exit 1 
fi  

if [ ! -z "$alidir" -a $stage -le 3 ]; then  
   echo "Checking accuracy of silence classifier using previously computed alignments"
   # Alignment dir specified: compare silences found with reference.
   silphonelist=`cat $lang/silphones.csl` || exit 1;
   n=`get_splits.pl $nj | awk '{print $1}'` # Just use first block for this.
   hyp="ark,s,cs:gunzip -c $dir/silprobs.$n.gz|"
   rhyp="$hyp reverse-weights ark:- ark:- |"
   ref="ark,s,cs:gunzip -c $alidir/$n.ali.gz | ali-to-post ark:- ark:- | weight-silence-post 0.0 $silphonelist $alidir/final.mdl ark:- ark:- | post-to-weights ark:- ark:- | reverse-weights ark:- ark:- |"
   rref="$ref reverse-weights ark:- ark:- |"
   s_s=`dot-weights "$ref" "$hyp" ark,t:- 2>/dev/null | awk '{x=x+$2} END{print x}'`
   s_ns=`dot-weights "$ref" "$rhyp" ark,t:- 2>/dev/null | awk '{x=x+$2} END{print x}'`
   ns_ns=`dot-weights "$rref" "$rhyp" ark,t:- 2>/dev/null | awk '{x=x+$2} END{print x}'`
   ns_s=`dot-weights "$rref" "$hyp" ark,t:- 2>/dev/null | awk '{x=x+$2} END{print x}'`

  ( echo "Reference  \ Hyp    Sil     Non-sil"
    echo " Sil                $s_s      $s_ns"
    echo "Non-sil             $ns_s     $ns_ns" ) | tee $dir/log/silence_accuracy.log


# Got this output in one case..
#Reference  \ Hyp    Sil     Non-sil
# Sil                99811.1      2314.9
#Non-sil             11488.8     475000


  # Also create sil.tacc and nonsil.tacc files, which are transition-accumulators
  # weighted by silence and non-silence probabilities respectively.  These
  # could be useful for creating more detailed diagnostics.

  ( gunzip -c $alidir/*.ali.gz | ali-to-post ark:- ark:- | \
    weight-post ark:- "ark,s,cs:gunzip -c $dir/silprobs.*.gz|" ark:- | \
    post-to-tacc --binary=false $alidir/final.mdl ark:- $dir/sil.tacc ) 2>$dir/log/sil_tacc.log || exit 1;

  ( gunzip -c $alidir/*.ali.gz | ali-to-post ark:- ark:- | \
    weight-post ark:- "ark,s,cs:gunzip -c $dir/silprobs.*.gz| reverse-weights ark:- ark:- |" ark:- | \
    post-to-tacc --binary=false $alidir/final.mdl ark:- $dir/nonsil.tacc ) 2>$dir/log/nonsil_tacc.log || exit 1;

fi

# Now, on the non-silence-only portion of the data, estimate fMLLR transforms.
# This is done in a two-model fashion, using the speaker-independent diagonal
# UBM to get the Gaussian alignments, and the speaker-dependent diagonal UBM
# to estimate the transforms with.

if [ $stage -le 4 ]; then
  echo "Computing fMLLR transforms (first pass)"
  for n in `get_splits.pl $nj`; do
    $cmd $dir/log/fmllr_pass1.$n.log \
      gmm-global-est-fmllr --spk2utt=ark:$data/split$nj/$n/spk2utt \
        --align-model=$ubmdir/final.si.dubm \
        "--gselect=ark,s,cs:gunzip -c $dir/gselect.$n.gz|" \
        "--weights=ark,s,cs:gunzip -c $dir/silprobs.$n.gz | reverse-weights ark:- ark:- |" \
        $ubmdir/final.dubm "${sifeatspart[$n]}" "ark:$dir/$n.pre_trans" \
       || touch $dir/.error &
  done
  wait
  [ -f $dir/.error ] && echo "Error computing fMLLR transforms (first pass)" && exit 1 
fi

if [ $stage -le 5 ]; then
  echo "Computing fMLLR transforms (second pass)"
  for n in `get_splits.pl $nj`; do
    feats="${sifeatspart[$n]} transform-feats --utt2spk=ark:$data/split$nj/$n/utt2spk ark,s,cs:$dir/$n.pre_trans ark:- ark:- |"
    $cmd $dir/log/fmllr_pass2.$n.log \
      gmm-global-est-fmllr --spk2utt=ark:$data/split$nj/$n/spk2utt \
        "--gselect=ark,s,cs:gunzip -c $dir/gselect.$n.gz|" \
        "--weights=ark,s,cs:gunzip -c $dir/silprobs.$n.gz | reverse-weights ark:- ark:- |" \
        $ubmdir/final.dubm "$feats" "ark:$dir/$n.tmp_trans" \
       || touch $dir/.error &
  done
  wait
  [ -f $dir/.error ] && echo "Error computing fMLLR transforms (second pass)" && exit 1 
  for n in `get_splits.pl $nj`; do
    compose-transforms --b-is-affine=true \
      ark:$dir/$n.tmp_trans ark:$dir/$n.pre_trans ark:$dir/$n.trans \
      2>$dir/log/compose.$n.log || exit 1;
  done
fi

rm $dir/*.{tmp_trans,pre_trans} 
rm $dir/gselect.*.gz
#rm $dir/silprobs.*.gz

exit 0
