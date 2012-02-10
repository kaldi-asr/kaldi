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
# Triphone model training, using cepstral mean normalization plus
# splice-9-frames, an LDA+MLLT transform, then speaker-specific
# affine transforms (fMLLR/CMLLR).  
#
# This training run starts from an initial directory that has
# alignments, models and transforms from an LDA+MLLT system:
#  ali, final.mdl, final.mat


if [ $# != 4 ]; then
   echo "Usage: steps/train_lda_mllt_sat.sh <data-dir> <lang-dir> <ali-dir> <exp-dir>"
   echo " e.g.: steps/train_lda_mllt_sat.sh data/train data/lang exp/tri2b_ali exp/tri3d"
   exit 1;
fi

if [ -f path.sh ]; then . path.sh; fi

data=$1
lang=$2
alidir=$3
dir=$4

scale_opts="--transition-scale=1.0 --acoustic-scale=0.1 --self-loop-scale=0.1"
realign_iters="5 10 15 20";  
# Previously had 2 4 6 12, but noticed very small objf impr on iter 4, so moved extra iters
# to later.
fmllr_iters="2 6 12 20";
silphonelist=`cat $lang/silphones.csl`
numiters=25    # Number of iterations of training
maxiterinc=15 # Last iter to increase #Gauss on.
numleaves=1800 # target num-leaves in tree building.
numgauss=$[$numleaves + $numleaves/2];  # starting num-Gauss.
     # Initially mix up to avg. 1.5 Gauss/state ( a bit more
     # than this, due to state clustering... then slowly mix 
     # up to final amount.
totgauss=9000 # Target #Gaussians
incgauss=$[($totgauss-$numgauss)/$maxiterinc] # per-iter increment for #Gauss
randprune=5.0 # for fMLLR accumulation, to speed it up (0.0 would be exact)

mkdir -p $dir
cp $alidir/final.mat $dir # Will use the same transform as in the baseline.


sifeats="ark:apply-cmvn --norm-vars=false --utt2spk=ark:$data/utt2spk ark:$alidir/cmvn.ark scp:$data/feats.scp ark:- | splice-feats ark:- ark:- | transform-feats $dir/final.mat ark:- ark:- |"
# This variable gets overwritten in this script:
feats="$sifeats"
cur_fmllr=

# compute integer form of transcripts.
scripts/sym2int.pl --ignore-first-field $lang/words.txt < $data/text > $dir/train.tra \
  || exit 1;

echo "Accumulating tree stats"

acc-tree-stats --ci-phones=$silphonelist $alidir/final.mdl "$feats" \
   ark:$alidir/ali $dir/treeacc 2> $dir/acc.tree.log  || exit 1;

echo "Computing questions for tree clustering"

cat $lang/phones.txt | awk '{print $NF}' | grep -v -w 0 > $dir/phones.list
cluster-phones $dir/treeacc $dir/phones.list $dir/questions.txt 2> $dir/questions.log || exit 1;
scripts/int2sym.pl $lang/phones.txt < $dir/questions.txt > $dir/questions_syms.txt
compile-questions $lang/topo $dir/questions.txt $dir/questions.qst 2>$dir/compile_questions.log || exit 1;

# Have to make silence root not-shared because we will not split it.
scripts/make_roots.pl --separate $lang/phones.txt $silphonelist shared split \
    > $dir/roots.txt 2>$dir/roots.log || exit 1;


echo "Building tree"
build-tree --verbose=1 --max-leaves=$numleaves \
    $dir/treeacc $dir/roots.txt \
    $dir/questions.qst $lang/topo $dir/tree  2> $dir/train_tree.log || exit 1;

gmm-init-model  --write-occs=$dir/1.occs  \
    $dir/tree $dir/treeacc $lang/topo $dir/1.mdl 2> $dir/init_model.log || exit 1;

gmm-mixup --mix-up=$numgauss $dir/1.mdl $dir/1.occs $dir/1.mdl \
   2>$dir/mixup.log || exit 1;

rm $dir/treeacc

# Convert alignments generated from monophone model, to be used as initial alignments.

convert-ali $alidir/final.mdl $dir/1.mdl $dir/tree ark:$alidir/ali ark:$dir/cur.ali 2>$dir/convert.log 
  # Debug step only: convert back and check they're the same.
  convert-ali $dir/1.mdl $alidir/final.mdl $alidir/tree ark:$dir/cur.ali ark:- \
   2>/dev/null | cmp - $alidir/ali || exit 1; 

# Make training graphs
echo "Compiling training graphs"
compile-train-graphs $dir/tree $dir/1.mdl  $lang/L.fst ark:$dir/train.tra \
    "ark:|gzip -c >$dir/graphs.fsts.gz"  2>$dir/compile_graphs.log  || exit 1;

x=1
while [ $x -lt $numiters ]; do
   echo Pass $x
   if echo $realign_iters | grep -w $x >/dev/null; then
     echo "Aligning data"
     gmm-align-compiled $scale_opts --beam=8 --retry-beam=40 $dir/$x.mdl \
             "ark:gunzip -c $dir/graphs.fsts.gz|" "$feats" \
             ark:$dir/cur.ali 2> $dir/align.$x.log || exit 1;
   fi
   if echo $fmllr_iters | grep -w $x >/dev/null; then
     echo "Estimating fMLLR"
    ( ali-to-post ark:$dir/cur.ali ark:- | \
      weight-silence-post 0.0 $silphonelist $dir/$x.mdl ark:- ark:- | \
      rand-prune-post $randprune ark:- ark:- | \
      gmm-est-fmllr --spk2utt=ark:$data/spk2utt $dir/$x.mdl "$feats" ark:- ark:$dir/tmp.trans ) \
      2>$dir/trans.$x.log || exit 1;
     if [ "$feats" == "$sifeats" ]; then # first time...
       mv $dir/tmp.trans $dir/cur.trans || exit 1;
       feats="$sifeats transform-feats --utt2spk=ark:$data/utt2spk ark:$dir/cur.trans ark:- ark:- |"
     else
       # compose the transforms; we apply tmp.trans after cur.trans so 
       # it comes first in matrix multiplication.
       compose-transforms --b-is-affine=true ark:$dir/tmp.trans ark:$dir/cur.trans \
          ark:$dir/tmp2.trans || exit 1;
       mv $dir/tmp2.trans $dir/cur.trans || exit 1;
       rm $dir/tmp.trans
     fi
   fi

   gmm-acc-stats-ali --binary=false $dir/$x.mdl "$feats" ark:$dir/cur.ali $dir/$x.acc 2> $dir/acc.$x.log || exit 1;
   gmm-est --write-occs=$dir/$[$x+1].occs --mix-up=$numgauss $dir/$x.mdl $dir/$x.acc $dir/$[$x+1].mdl 2> $dir/update.$x.log || exit 1;
   rm $dir/$x.mdl $dir/$x.acc
   rm $dir/$x.occs 
   if [[ $x -le $maxiterinc ]]; then 
      numgauss=$[$numgauss+$incgauss];
   fi
   x=$[$x+1];
done

# Accumulate stats for "alignment model" which is as the model but with
# unadapted features.
( ali-to-post ark:$dir/cur.ali ark:-  | \
  gmm-acc-stats-twofeats $dir/$x.mdl "$feats" "$sifeats" ark:- $dir/$x.acc2 ) \
    2>$dir/acc_alimdl.log || exit 1;
  # Update model.
gmm-est --write-occs=$dir/final.occs --remove-low-count-gaussians=false $dir/$x.mdl $dir/$x.acc2 $dir/$x.alimdl \
      2>$dir/est_alimdl.log  || exit 1;
rm $dir/$x.acc2

( cd $dir; rm final.{mdl,alimdl,occs} 2>/dev/null; ln -s $x.mdl final.mdl; 
  ln -s $x.alimdl final.alimdl; ln -s $x.occs final.occs; )

echo Done
