#!/bin/bash
# Copyright 2010-2011 Microsoft Corporation  Arnab Ghoshal
#                     Univ. Erlangen-Nuremberg  Korbinian Riedhammer

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
# Triphone model training, using delta-delta features and cepstral
# mean normalization.  It starts from an existing directory (e.g.
# exp/mono), supplied as an argument, which is assumed to be built using
# the same type of features.

# this training scripts assumes tri1 alignments to start from

if [[ $# != 4 && $# != 10 ]]; then
   echo "Usage: steps/train_2lvl_diag.sh <data-dir> <lang-dir> <ali-dir> <exp-dir> [ <num-codebooks> <num-gaussians> <num-tree-leaves> <smooth-type> <tau> <rho> ]"
   echo " e.g.: steps/train_2lvl_diag.sh data/train data/lang exp/tri1_ali exp/tri1-2lvl-default"
   echo " e.g.: steps/train_2lvl_diag.sh data/train data/lang exp/tri1_ali exp/tri1-2lvl 100 1024 1800 0 10 0.5"
   exit 1;
fi

if [ -f path.sh ]; then . path.sh; fi

data=$1
lang=$2
alidir=$3
dir=$4

if [ ! -f $alidir/final.mdl -o ! -f $alidir/ali ]; then
  echo "Error: alignment dir $alidir does not contain final.mdl and ali"
  exit 1;
fi

scale_opts="--transition-scale=1.0 --acoustic-scale=0.1 --self-loop-scale=0.1"
realign_iters="5 10 15 20";  
silphonelist=`cat $lang/silphones.csl`
numiters=25          # Number of iterations of training

# default parameters
max_leaves_first=78
max_leaves_second=2500
totgauss=2048
smooth_type=0
tau=0
rho=0.5
mingauss=3           # minimum size of codebook
power=1

inter=""
preserve_counts=""

if [ $# == 10 ]; then
  max_leaves_first=$5  # Number of codebooks
  max_leaves_second=$7 # target num-leaves in tree building.
  totgauss=$6          # Target total #Gaussians in codebooks
  smooth_type=$8       # (0, regular Interpolate1) (1, preserve-counts, Interpolate2)
  tau=$9               # set to > 0 to activate smoothing of suff. stats. (weights only)
  rho=${10}            # set to > 0 to actiavte smoothing of new model with prior model (weights only)
fi

# interpolation between iterations requested?
if [ "$rho" != "0" ]; then
  inter="--interpolation-weight=$rho --interpolate-weights"
fi

# smothing of stats requested?
if [ "$smooth_type" == "0" ]; then
  preserve_counts=""
elif [ "$smooth_type" == "1" ]; then
  preserve_counts="--preserve-counts"
else
  echo "Invalid smoothing type $smooth_type"
  exit 1
fi

mkdir -p $dir

# write out the params we're using, just for the record
echo "Training started: `date`
$0 $@
max_leaves_first=$max_leaves_first
max_leaves_second=$max_leaves_second
totgauss=$totgauss
smooth_type=$smooth_type
tau=$tau
rho=$rho
intra=$intra
preserve_counts=$preserve_counts" > $dir/.params

feats="ark:apply-cmvn --norm-vars=false --utt2spk=ark:$data/utt2spk ark:$alidir/cmvn.ark scp:$data/feats.scp ark:- | add-deltas ark:- ark:- |"

# compute integer form of transcripts.
scripts/sym2int.pl --ignore-first-field $lang/words.txt < $data/text > $dir/train.tra \
  || exit 1;

echo "Accumulating tree stats"
acc-tree-stats  --ci-phones=$silphonelist $alidir/final.mdl "$feats" \
   ark:$alidir/ali $dir/treeacc 2> $dir/acc.tree.log  || exit 1;

echo "Computing questions for tree clustering"
cat $lang/phones.txt | awk '{print $NF}' | grep -v -w 0 > $dir/phones.list
cluster-phones $dir/treeacc $dir/phones.list $dir/questions.txt 2> $dir/questions.log || exit 1;
scripts/int2sym.pl $lang/phones.txt < $dir/questions.txt > $dir/questions_syms.txt
compile-questions $lang/topo $dir/questions.txt $dir/questions.qst 2>$dir/compile_questions.log || exit 1;

# Have to make silence root not-shared because we will not split it.
scripts/make_roots.pl --separate $lang/phones.txt $silphonelist shared split \
    > $dir/roots.txt 2>$dir/roots.log || exit 1;

# build the 2-lvl tree, make sure to not cluster the leaves!
echo "Building tree"
build-tree-two-level --verbose=1 --cluster-leaves=false \
    --max-leaves_first=$max_leaves_first \
    --max-leaves_second=$max_leaves_second \
    $dir/treeacc $dir/roots.txt \
    $dir/questions.qst $lang/topo $dir/tree $dir/tree.map 2> $dir/train_tree.log || exit 1;

# codebook initialization as desired...
echo "Initializing codebooks"
init-tied-codebooks --power=$power --split-gaussians=true --diag=true --min-gauss=$mingauss --max-gauss=$totgauss \
    $dir/tree $dir/treeacc $dir/ubm-diag $dir/tree.map 2> $dir/init-tied-codebooks.err || exit 1;

rm $dir/treeacc

echo "Initializing model"
if [ $max_leaves_first -gt 99999 ]; then
  echo "Too many codebooks, adjust training script"
  exit 1
elif [ $max_leaves_first -gt 9999 ]; then
  tied-diag-gmm-init-model $dir/tree $lang/topo $dir/tree.map $dir/ubm-diag.{?,??,???,????,?????} $dir/1.mdl 2> $dir/init_model.log || exit 1;
elif [ $max_leaves_first -gt 999 ]; then
  tied-diag-gmm-init-model $dir/tree $lang/topo $dir/tree.map $dir/ubm-diag.{?,??,???,????} $dir/1.mdl 2> $dir/init_model.log || exit 1;
elif [ $max_leaves_first -gt 99 ]; then
  tied-diag-gmm-init-model $dir/tree $lang/topo $dir/tree.map $dir/ubm-diag.{?,??,???} $dir/1.mdl 2> $dir/init_model.log || exit 1;
elif [ $max_leaves_first -gt 9 ]; then
  tied-diag-gmm-init-model $dir/tree $lang/topo $dir/tree.map $dir/ubm-diag.{?,??} $dir/1.mdl 2> $dir/init_model.log || exit 1;
else
  tied-diag-gmm-init-model $dir/tree $lang/topo $dir/tree.map $dir/ubm-diag.? $dir/1.mdl 2> $dir/init_model.log || exit 1;
fi

# we don't need these any more
rm $dir/ubm-diag.*

# Convert alignments generated from cont/triphone model, to use as initial alignments.

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
     tied-diag-gmm-align-compiled $scale_opts --beam=8 --retry-beam=40 $dir/$x.mdl \
             "ark:gunzip -c $dir/graphs.fsts.gz|" "$feats" \
             ark:$dir/cur.ali 2> $dir/align.$x.log || exit 1;
   fi
   
   tied-diag-gmm-acc-stats-ali --binary=false $dir/$x.mdl "$feats" ark:$dir/cur.ali $dir/$x.acc 2> $dir/acc.$x.log  || exit 1;

   # suff. stats smoothing?
   if [ "$tau" != "0" ]; then
	 smooth-stats-diag $ssmoothing --tau=$tau $dir/tree $dir/tree.map $dir/$x.acc $dir/$x.acc.tmp 2> $dir/smooth.$x.err > $dir/smooth.$x.out || exit 1;
	 mv $dir/$x.acc.tmp $dir/$x.acc
   fi

   tied-diag-gmm-est $intra --write-occs=$dir/$[$x+1].occs $dir/$x.mdl $dir/$x.acc $dir/$[$x+1].mdl 2> $dir/update.$x.log || exit 1;
   
   rm $dir/$x.mdl $dir/$x.acc
   rm $dir/$x.occs
   x=$[$x+1];
done

( cd $dir; rm final.mdl 2>/dev/null; ln -s $x.mdl final.mdl; ln -s $x.occs final.occs )

echo Done
