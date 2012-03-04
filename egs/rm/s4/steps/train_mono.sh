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
# Flat start and monophone training, with delta-delta features.
# This script applies cepstral mean normalization (per speaker),
# unlike the corresponding script in s1/

if [ $# != 3 ]; then
   echo "Usage: steps/train_mono.sh <data-dir> <lang-dir> <exp-dir>"
   echo " e.g.: steps/train_mono.sh data/train.1k data/lang exp/mono"
   exit 1;
fi


data=$1
lang=$2
dir=$3

if [ -f path.sh ]; then . path.sh; fi

# Configuration:
scale_opts="--transition-scale=1.0 --acoustic-scale=0.1 --self-loop-scale=0.1"
numiters=30    # Number of iterations of training
maxiterinc=20 # Last iter to increase #Gauss on.
numgauss=250 # Initial num-Gauss (must be more than #states=3*phones).
totgauss=1000 # Target #Gaussians.  
incgauss=$[($totgauss-$numgauss)/$maxiterinc] # per-iter increment for #Gauss
realign_iters="1 2 3 4 5 6 7 8 9 10 12 15 20 25";

mkdir -p $dir
echo "Computing cepstral mean and variance statistics"

compute-cmvn-stats  scp:$data/feats.scp ark:$dir/cmvn.ark 2>$dir/cmvn.log || exit 1;

feats="ark:apply-cmvn --norm-vars=false ark:$dir/cmvn.ark scp:$data/feats.scp ark:- | add-deltas ark:- ark:- |"

# compute integer form of transcripts.
scripts/sym2int.pl --ignore-first-field $lang/words.txt < $data/text > $dir/train.tra \
  || exit 1;

echo "Initializing monophone system."

gmm-init-mono "--train-feats=$feats subset-feats --n=10 ark:- ark:-|" $lang/topo 39  \
   $dir/0.mdl $dir/tree 2> $dir/init.log || exit 1;


echo "Compiling training graphs"
compile-train-graphs $dir/tree $dir/0.mdl  $lang/L.fst \
  ark:$dir/train.tra  "ark:|gzip -c >$dir/graphs.fsts.gz"  \
  2>$dir/compile_graphs.log || exit 1 

echo Pass 0

align-equal-compiled "ark:gunzip -c $dir/graphs.fsts.gz|" "$feats" \
   ark,t,f:-  2>$dir/align.0.log | \
 gmm-acc-stats-ali --binary=true $dir/0.mdl "$feats" ark:- \
     $dir/0.acc 2> $dir/acc.0.log  || exit 1;

# In the following steps, the --min-gaussian-occupancy=3 option is important, otherwise
# we fail to est "rare" phones and later on, they never align properly.

gmm-est --min-gaussian-occupancy=3  --mix-up=$numgauss \
    $dir/0.mdl $dir/0.acc $dir/1.mdl 2> $dir/update.0.log || exit 1;

rm $dir/0.acc

beam=4 # will change to 8 below after 1st pass
x=1
while [ $x -lt $numiters ]; do
  echo "Pass $x"
  if echo $realign_iters | grep -w $x >/dev/null; then
    echo "Aligning data"
    gmm-align-compiled $scale_opts --beam=$beam --retry-beam=$[$beam*4] $dir/$x.mdl \
        "ark:gunzip -c $dir/graphs.fsts.gz|" "$feats" t,ark:$dir/cur.ali \
        2> $dir/align.$x.log || exit 1;
  fi
  gmm-acc-stats-ali --binary=false $dir/$x.mdl "$feats" ark:$dir/cur.ali $dir/$x.acc 2> $dir/acc.$x.log  || exit 1;
  gmm-est --write-occs=$dir/$[$x+1].occs --mix-up=$numgauss $dir/$x.mdl $dir/$x.acc $dir/$[$x+1].mdl 2> $dir/update.$x.log || exit 1;
  rm $dir/$x.mdl $dir/$x.acc $dir/$x.occs 2>/dev/null
  if [ $x -le $maxiterinc ]; then
     numgauss=$[$numgauss+$incgauss];
  fi
  beam=8
  x=$[$x+1]
done

( cd $dir; rm final.mdl 2>/dev/null; ln -s $x.mdl final.mdl; ln -s $x.occs final.occs )

# example of showing the alignments:
# show-alignments data/lang/phones.txt $dir/30.mdl ark:$dir/cur.ali | head -4

