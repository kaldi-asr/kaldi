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

# train_et2.sh is as train_et.sh but using an adapt model with
# fewer Gaussians.  Seeing if this makes the warp distribution more
# bimodal.



if [ -f path.sh ]; then . path.sh; fi
srcdir=exp/adapt2
dir=exp/et2
srcmodel=$srcdir/20.mdl

normtype=mean # could be mean or none or mean-and-var

spk2utt_opt=--spk2utt=ark:$dir/spk2utt
utt2spk_opt=--utt2spk=ark:$dir/utt2spk
# for per-utterance, uncomment the following [this would make it worse]:
# spk2utt_opt=
# utt2spk_opt=
feats="ark:add-deltas scp:$dir/train.scp ark:- |"

mkdir -p $dir

nspk=109 # Use all 109 RM training speakers.
nutt=15 # Use at most 15 utterances from each speaker.

head -$nspk data/train.spk2utt | \
   awk '{ printf("%s ",$1); for(x=2; x<=NF&&x<='$nutt'+1;x++)
         {  printf("%s ", $x); } printf("\n"); }' > $dir/spk2utt

scripts/spk2utt_to_utt2spk.pl < $dir/spk2utt > $dir/utt2spk
cat $dir/utt2spk | awk '{print $1}' > $dir/uttlist
scripts/filter_scp.pl $dir/uttlist <data/train.scp >$dir/train.scp

silphonelist=`cat data/silphones.csl`

cp $srcdir/tree $dir
cp $srcdir/phone_map $dir

# Use a subset of a training utts from srcdir, so we use the alignments from there:
# link these.
( 
  cd $dir
  ln -s ../../$srcdir/cur.ali .
  ln -s ../../$srcmodel 0.mdl
)

# Init the transform:

gmm-init-et --normalize-type=$normtype --binary=false --dim=39 $dir/0.et 2>$dir/init_et.log || exit 1

  
for x in 0 1 2 3 4 5 6 7 8 9 10 11; do
    x1=$[$x+1]; 

    # Work out current transforms:
   ( ali-to-post ark:$dir/cur.ali ark:- | \
    weight-silence-post 0.0 $silphonelist $srcmodel ark:- ark:- | \
    gmm-post-to-gpost $srcmodel "$feats" ark:- ark:- | \
    gmm-est-et $spk2utt_opt --verbose=1 $dir/$x.mdl $dir/$x.et "$feats" ark:- ark:$dir/$x.trans ark,t:$dir/$x.warp ) 2> $dir/trans.$x.log || exit 1;

    # Accumulate stats to update model:
   ( transform-feats $utt2spk_opt ark:$dir/$x.trans "$feats" ark:- 2>$dir/apply_fmllr.$x.log | \
    gmm-acc-stats-twofeats $srcmodel "$feats" ark:- "ark:cat $dir/cur.ali | ali-to-post ark:- ark:- |" $dir/$x.acc ) 2>$dir/gmm_acc.$x.log || exit 1;


    # Check likelihoods (must add the fMLLR determinants from apply_fmllr.$x.log, to get meaningful
    # figures.)
    ( transform-feats $utt2spk_opt ark:$dir/$x.trans "$feats" ark:-  | \
     gmm-acc-stats $dir/$x.mdl ark:- "ark:cat $dir/cur.ali | ali-to-post ark:- ark:- |" /dev/null ) 2>$dir/gmm_getlike.$x.log || exit 1;


    gmm-est --remove-low-count-gaussians=false $dir/$x.mdl $dir/$x.acc $dir/$x1.mdl 2>$dir/gmm_est.$x.log || exit 1;

    # Next estimate either A or B, depending on iteration:
    if [ $[$x%2] == 0 ]; then  # Estimate A:
    ( ali-to-post ark:$dir/cur.ali ark:- | \
      weight-silence-post 0.0 $silphonelist $srcmodel ark:- ark:- | \
      gmm-post-to-gpost $srcmodel "$feats" ark:- ark:- | \
      gmm-et-acc-a $spk2utt_opt --verbose=1 $dir/$x1.mdl $dir/$x.et "$feats" ark:- $dir/$x.et_acc_a ) 2> $dir/acc_a.$x.log || exit 1;
      gmm-et-est-a --verbose=1 $dir/$x.et $dir/$x1.et $dir/$x.et_acc_a 2> $dir/update_a.$x.log || exit 1;
      rm $dir/$x.et_acc_a
    else
    ( ali-to-post ark:$dir/cur.ali ark:- | \
      weight-silence-post 0.0 $silphonelist $srcmodel ark:- ark:- | \
      gmm-post-to-gpost $srcmodel "$feats" ark:- ark:- | \
      gmm-et-acc-b $spk2utt_opt --verbose=1 $dir/$x1.mdl $dir/$x.et "$feats" ark:- ark:$dir/$x.trans ark:$dir/$x.warp $dir/$x.et_acc_b 2> $dir/acc_b.$x.log || exit 1;
      gmm-et-est-b --verbose=1 $dir/$x.et $dir/$x1.et $dir/$x.mat $dir/$x.et_acc_b ) 2> $dir/update_b.$x.log || exit 1;
      rm $dir/$x.et_acc_b
      # Careful!: gmm-transform-means here changes $x1.mdl in-place. 
      gmm-transform-means $dir/$x.mat $dir/$x1.mdl $dir/$x1.mdl 2> $dir/transform_means.$x.log
    fi
    rm $dir/$x.trans 
    if [ $x != 0 ]; then
      rm $dir/$x.mdl  # keep 0.mdl as it's the alignment model.
    fi
    rm $dir/$x.acc
    x=$[$x+1];
done

for n in 0 1 2 3 4 5 6 7 8 9 10 11; do
 cat $dir/$n.warp | scripts/process_warps.pl data/spk2gender.map > $dir/warps.$n
done
