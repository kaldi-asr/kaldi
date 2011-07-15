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
if [ -f path.sh ]; then . path.sh; fi

# Train the monophone on a subset-- no point using all the data.
dir=exp/mono
n=1000
feats="ark:add-deltas --print-args=false scp:$dir/train.scp ark:- |"
# need to quote when passing as an argument, as in "$feats",
# since it has spaces in it.
scale_opts="--transition-scale=1.0 --acoustic-scale=0.1 --self-loop-scale=0.1"

numiters=30    # Number of iterations of training
maxiterinc=20 # Last iter to increase #Gauss on.
numgauss=250 # Initial num-Gauss (must be more than #states=3*phones).
totgauss=1000 # Target #Gaussians.  
incgauss=$[($totgauss-$numgauss)/$maxiterinc] # per-iter increment for #Gauss
realign_iters="1 2 3 4 5 6 7 8 9 10 12 15 20 25";


mkdir -p $dir
scripts/subset_scp.pl $n data/train.scp > $dir/train.scp


silphones=`cat data/silphones.csl | sed 's/:/ /g'`
nonsilphones=`cat data/nonsilphones.csl | sed 's/:/ /g'`
cat conf/topo.proto | sed "s:NONSILENCEPHONES:$nonsilphones:" | sed "s:SILENCEPHONES:$silphones:" > $dir/topo

gmm-init-mono '--train-feats=ark:head -10 data/train.scp | add-deltas scp:- ark:- |' $dir/topo 39  $dir/0.mdl $dir/tree 2> $dir/init.log || exit 1;


echo "Compiling training graphs"
compile-train-graphs $dir/tree $dir/0.mdl  data/L.fst \
       "ark:scripts/subset_scp.pl $n data/train.tra|" \
   "ark:|gzip -c >$dir/graphs.fsts.gz"  2>$dir/compile_graphs.log || exit 1 

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
  gmm-est --mix-up=$numgauss $dir/$x.mdl $dir/$x.acc $dir/$[$x+1].mdl 2> $dir/update.$x.log || exit 1;
  rm $dir/$x.mdl $dir/$x.acc
  if [ $x -le $maxiterinc ]; then
     numgauss=$[$numgauss+$incgauss];
  fi
  beam=8
  x=$[$x+1]
done

( cd $dir; rm final.mdl 2>/dev/null; ln -s $x.mdl final.mdl )

# example of showing the alignments:
# show-alignments data/phones.txt $dir/30.mdl ark:$dir/cur.ali | head -4

