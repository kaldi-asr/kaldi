#!/bin/bash
# Copyright 2010-2012 Microsoft Corporation  Arnab Ghoshal  Johns Hopkins University (Author: Daniel Povey)

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

# This script trains SGMMs, but the tree is trained using the SGMM-type
# tree stats.

# Old comment describing previous script:
# Trains SGMM on top of LDA plus [something] features, where the [something]
# might be e.g. MLLT, or some kind of speaker-specific transform.  This version
# of the script trains the tree in an integrated way, with the SGMM itself.  It
# first trains an initial SGMM for a few iterations, with a single sub-state per
# state, based on the old tree.  It then accumulates "SGMM tree stats" using
# this SGMM, and does the clustering with this; the tree building program
# outputs an SGMM that's compatible with the new tree.


if [ $# != 5 ]; then
   echo "Usage: steps/train_sgmm_lda_etc_2.sh <data-dir> <lang-dir> <ali-dir> <ubm> <exp-dir>"
   echo " e.g.: steps/train_sgmm_lda_etc_2.sh data/train data/lang exp/tri2b_ali exp/ubm3c/final.ubm exp/sgmm3d"
   exit 1;
fi

if [ -f path.sh ]; then . path.sh; fi

# This is SGMM with speaker vectors, on top of LDA+STC/MLLT features.
# To be run from ..

simple_init=true 
data=$1
lang=$2
alidir=$3
ubm=$4
dir=$5

mkdir -p $dir || exit 1;
cp $alidir/final.mat $dir/final.mat || exit 1;

scale_opts="--transition-scale=1.0 --acoustic-scale=0.1 --self-loop-scale=0.1"

numpreiters=5 # Number of iterations for the initial phases of SGMM
   # training before we build the tree.
numiters=25   # Total number of iterations

realign_iters="5 10 15"; 
spkvec_iters="5 8 12 17 22"
silphonelist=`cat $lang/silphones.csl`
spkspacedim=40
phnspacedim=41
numleaves=2500
numsubstates=2500 # Initial #-substates.
totsubstates=7500 # Target #-substates.
maxiterinc=15 # Last iter to increase #substates on.
incsubstates=$[($totsubstates-$numsubstates)/$maxiterinc] # per-iter increment for #substates
gselect_opt="--gselect=ark,s,cs:gunzip -c $dir/gselect.gz|"
# Initially don't have speaker vectors, but change this after
# we estimate them.
spkvecs_opt=
randprune=0.1
mkdir -p $dir

utt2spk_opt="--utt2spk=ark:$data/utt2spk"
spk2utt_opt="--spk2utt=ark:$data/spk2utt"

feats="ark,s,cs:apply-cmvn --norm-vars=false --utt2spk=ark:$data/utt2spk ark:$alidir/cmvn.ark scp:$data/feats.scp ark:- | splice-feats ark:- ark:- | transform-feats $dir/final.mat ark:- ark:- |"

if [ -f $alidir/trans.ark ]; then
   echo "Running with speaker transforms $alidir/trans.ark"
   feats="$feats transform-feats --utt2spk=ark:$data/utt2spk ark:$alidir/trans.ark ark:- ark:- |"
fi

if [ ! -f $ubm ]; then
  echo "No UBM in $ubm"
  exit 1;
fi

# Create an SGMM based on the old tree.  We'll estimate this for a few iterations
# before creating our new tree.
sgmm-init --spk-space-dim=$spkspacedim --phn-space-dim=$phnspacedim $lang/topo $alidir/tree $ubm \
    $dir/0.pre_mdl 2> $dir/init_sgmm.log || exit 1;

# Compute the Gaussian-selection info.
if [ ! -f $dir/gselect.gz ]; then
 sgmm-gselect $dir/0.pre_mdl "$feats" ark,t:- 2>$dir/gselect.log | \
    gzip -c > $dir/gselect.gz || exit 1;
fi

if [ ! -f $dir/$numpreiters.pre_mdl ]; then 

# Update this model for a few iterations.  We won't bother with speaker adaptation yet.
# we can add this later.
  iter=0
  while [ $iter -lt $numpreiters ]; do
    echo "Pre-iter $iter ... "
    if [ $iter -eq 0 ]; then
      flags=vwcSt
    else
      flags=vMwcSt
    fi
    sgmm-acc-stats --update-flags=$flags "$gselect_opt" --rand-prune=$randprune --binary=false $dir/$iter.pre_mdl "$feats" "ark:ali-to-post ark:$alidir/ali ark:-|" $dir/$iter.acc 2> $dir/pre_acc.$iter.log  || exit 1;
    sgmm-est --update-flags=$flags $dir/$iter.pre_mdl $dir/$iter.acc $dir/$[$iter+1].pre_mdl 2> $dir/pre_update.$iter.log || exit 1;

    rm $dir/$iter.pre_mdl $dir/$iter.acc
    iter=$[$iter+1];
  done
fi

# We rebuild the tree because we want a larger #states than for a normal
# GMM system (the optimum #states for SGMMs tends to be a bit higher).

if [ ! -f $dir/streeacc ]; then
  sgmm-acc-tree-stats "$gselect_opt" --ci-phones=$silphonelist \
    $dir/$numpreiters.pre_mdl "$feats" ark:$alidir/ali \
    $dir/streeacc 2> $dir/acc.tree.log || exit 1;
fi


if [ ! -f $dir/roots.txt ]; then
  cat $lang/phones.txt | awk '{print $NF}' | grep -v -w 0 > $dir/phones.list
  sgmm-cluster-phones $dir/$numpreiters.pre_mdl $dir/streeacc $dir/phones.list $dir/questions.txt 2> $dir/questions.log || exit 1;
  scripts/int2sym.pl $lang/phones.txt < $dir/questions.txt > $dir/questions_syms.txt
  compile-questions $lang/topo $dir/questions.txt $dir/questions.qst 2>$dir/compile_questions.log || exit 1;
  scripts/make_roots.pl --separate $lang/phones.txt $silphonelist shared split > $dir/roots.txt 2>$dir/roots.log || exit 1;
fi

if [ ! -f $dir/tree ]; then
  # This program build the tree and also initializes the SGMM.
  sgmm-build-tree --verbose=1 --max-leaves=$numleaves \
    $dir/$numpreiters.pre_mdl $dir/streeacc $dir/roots.txt \
    $dir/questions.qst $dir/tree  2> $dir/train_tree.log || exit 1;
fi

if $simple_init; then
  sgmm-init --spk-space-dim=$spkspacedim --phn-space-dim=$phnspacedim $lang/topo $dir/tree $ubm \
     $dir/0.mdl 2> $dir/init_sgmm_final.log || exit 1;
else
  sgmm-init-from-tree-stats $dir/$numpreiters.pre_mdl \
    $dir/tree $dir/streeacc $dir/0.mdl 2>$dir/init_sgmm_final.log || exit 1;
fi

convert-ali $alidir/final.mdl $dir/0.mdl $dir/tree ark:$alidir/ali \
    ark:$dir/cur.ali 2>$dir/convert.log 

# Make training graphs
echo "Compiling training graphs"

compile-train-graphs $dir/tree $dir/0.mdl $lang/L.fst  \
  "ark:scripts/sym2int.pl --ignore-first-field $lang/words.txt < $data/text |" \
  "ark:|gzip -c >$dir/graphs.fsts.gz"  2>$dir/compile_graphs.log  || exit 1 


iter=0
while [ $iter -lt $numiters ]; do
   echo "Pass $iter ... "
   if echo $realign_iters | grep -w $iter >/dev/null; then
      echo "Aligning data"
      sgmm-align-compiled $spkvecs_opt $utt2spk_opt $scale_opts "$gselect_opt" \
          --beam=8 --retry-beam=40 $dir/$iter.mdl \
          "ark:gunzip -c $dir/graphs.fsts.gz|" "$feats" \
          ark:$dir/cur.ali 2> $dir/align.$iter.log || exit 1;
   fi
   if echo $spkvec_iters | grep -w $iter >/dev/null; then
    ( ali-to-post ark:$dir/cur.ali ark:- | \
      weight-silence-post 0.01 $silphonelist $dir/$iter.mdl ark:- ark:- | \
      sgmm-est-spkvecs $spk2utt_opt $spkvecs_opt "$gselect_opt" \
        --rand-prune=$randprune $dir/$iter.mdl \
       "$feats" ark,s,cs:- ark:$dir/tmp.vecs ) 2>$dir/spkvecs.$iter.log || exit 1;
      mv $dir/tmp.vecs $dir/cur.vecs
      spkvecs_opt="--spk-vecs=ark:$dir/cur.vecs"
   fi  
   if [ $[$iter%2] -eq 1 -a $iter -gt 4 ]; then # even iters after 4 (i.e. starting from 6)...
     flags=vNwcSt
   else
     flags=vMwcSt
   fi
   sgmm-acc-stats $spkvecs_opt $utt2spk_opt --update-flags=$flags "$gselect_opt" --rand-prune=$randprune --binary=false $dir/$iter.mdl "$feats" "ark:ali-to-post ark:$dir/cur.ali ark:-|" $dir/$iter.acc 2> $dir/acc.$iter.log  || exit 1;
   sgmm-est --update-flags=$flags --split-substates=$numsubstates --write-occs=$dir/$[$iter+1].occs $dir/$iter.mdl $dir/$iter.acc $dir/$[$iter+1].mdl 2> $dir/update.$iter.log || exit 1;

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
  ln -s $iter.mdl final.mdl; 
  ln -s $iter.alimdl final.alimdl;
  ln -s $iter.occs final.occs )

