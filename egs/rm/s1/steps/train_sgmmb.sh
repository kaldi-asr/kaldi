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


if [ -f path.sh ]; then . path.sh; fi

# To be run from ..
# You must run init_sgmma.sh first.
# We rely on the initial model exp/sgmma/0.mdl being there

dir=exp/sgmmb
srcdir=exp/tri1
srcmodel=$srcdir/final.mdl
srcgraphs="ark:gunzip -c $srcdir/graphs.fsts.gz|"
scale_opts="--transition-scale=1.0 --acoustic-scale=0.1 --self-loop-scale=0.1"

numiters=25   # Total number of iterations

ubm=exp/ubma/4.ubm
realign_iters="5 10 15"; 
spkvec_iters="5 8 12 17 22"
silphonelist=`cat data/silphones.csl`
numsubstates=1500 # Initial #-substates.
totsubstates=5000 # Target #-substates.
maxiterinc=15 # Last iter to increase #substates on.
incsubstates=$[($totsubstates-$numsubstates)/$maxiterinc] # per-iter increment for #substates
gselect_opt="--gselect=ark:gunzip -c $dir/gselect.gz|"
# Initially don't have speaker vectors, but change this after
# we estimate them.
spkvecs_opt=
randprune=0.1
mkdir -p $dir

utt2spk_opt="--utt2spk=ark:data/train.utt2spk"
spk2utt_opt="--spk2utt=ark:data/train.spk2utt"
feats="ark:add-deltas --print-args=false scp:data/train.scp ark:- |"

if [ ! -f $ubm ]; then
  echo "No UBM in $ubm"
fi

sgmm-init --spk-space-dim=39 $srcdir/final.mdl $ubm $dir/0.mdl 2> $dir/sgmm_init.log || exit 1;

cp $srcdir/tree $dir

echo "aligning all training data"
if [ ! -f $dir/0.ali ]; then
  gmm-align-compiled  $scale_opts --beam=8 --retry-beam=40  $srcmodel "$srcgraphs" \
        "$feats" ark,t:$dir/0.ali 2> $dir/align.0.log || exit 1;
fi

if [ ! -f $dir/0.mdl ]; then
   echo "you must run init_sgmm.sh before train_sgmm1.sh"
   exit 1
fi

if [ ! -f $dir/gselect.gz ]; then
 sgmm-gselect $dir/0.mdl "$feats" ark,t:- 2>$dir/gselect.log | gzip -c > $dir/gselect.gz || exit 1;
fi

cp $dir/0.ali $dir/cur.ali || exit 1;

iter=0
while [ $iter -lt $numiters ]; do
   echo "Pass $iter ... "
   if echo $realign_iters | grep -w $iter >/dev/null; then
      echo "Aligning data"
      sgmm-align-compiled $spkvecs_opt $utt2spk_opt $scale_opts "$gselect_opt" \
         --beam=8 --retry-beam=40 $dir/$iter.mdl "$srcgraphs" "$feats" \
      	ark:$dir/cur.ali 2> $dir/align.$iter.log || exit 1;
   fi
   if echo $spkvec_iters | grep -w $iter >/dev/null; then
    ( ali-to-post ark:$dir/cur.ali ark:- | \
      weight-silence-post 0.01 $silphonelist $dir/$iter.mdl ark:- ark:- | \
      sgmm-est-spkvecs $spk2utt_opt $spkvecs_opt "$gselect_opt" \
        --rand-prune=$randprune $dir/$iter.mdl \
       "$feats" ark:- ark:$dir/cur.vecs  2>$dir/spkvecs.$iter.log ) || exit 1;
      spkvecs_opt="--spk-vecs=ark:$dir/cur.vecs"
   fi  
   if [ $iter -eq 0 ]; then
     flags=vwcS
   elif [ $[$iter%2] -eq 1 -a $iter -gt 4 ]; then # even iters after 4...
     flags=vNwcS
   else
     flags=vMwcS
   fi
   if [ ! -f $dir/$[$iter+1].mdl ]; then
     sgmm-acc-stats $spkvecs_opt $utt2spk_opt --update-flags=$flags "$gselect_opt" --rand-prune=$randprune --binary=false $dir/$iter.mdl "$feats" "ark:ali-to-post ark:$dir/cur.ali ark:-|" $dir/$iter.acc 2> $dir/acc.$iter.log  || exit 1;
     sgmm-est --update-flags=$flags --split-substates=$numsubstates --write-occs=$dir/$[$iter+1].occs $dir/$iter.mdl $dir/$iter.acc $dir/$[$iter+1].mdl 2> $dir/update.$iter.log || exit 1;
   fi
  	rm $dir/$iter.mdl $dir/$iter.acc
  	rm $dir/$iter.occs 
    if [ $iter -lt $maxiterinc ]; then
       numsubstates=$[$numsubstates+$incsubstates]
    fi
    iter=$[$iter+1];
done


# The point of this last phase of accumulation is to get Gaussian-level
# alignments with the speaker vectors but accumulate stats without
# any speaker vectors; we re-estimate M, w, c and S to get a model
# that's compatible with not having speaker vectors.


flags=MwcS
( ali-to-post ark:$dir/cur.ali ark:- | \
  sgmm-post-to-gpost $spkvecs_opt $utt2spk_opt "$gselect_opt" \
                  $dir/$iter.mdl "$feats" ark,s,cs:- ark:- | \
  sgmm-acc-stats-gpost --update-flags=$flags  $dir/$iter.mdl "$feats" \
            ark,s,cs:- $dir/$iter.aliacc ) 2> $dir/acc_ali.$iter.log || exit 1;
sgmm-est --update-flags=$flags --remove-speaker-space=true $dir/$iter.mdl \
    $dir/$iter.aliacc $dir/$iter.alimdl 2>$dir/update_ali.$iter.log || exit 1;


( cd $dir; rm final.mdl final.occs 2>/dev/null; 
  ln -s $iter.mdl final.mdl; ln -s $iter.alimdl final.alimdl;
  ln -s $iter.occs final.occs )
