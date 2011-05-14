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

dir=exp/sgmm
srcdir=exp/tri1
srcmodel=$srcdir/final.mdl
srcgraphs="ark:gunzip -c $srcdir/graphs.fsts.gz|"
scale_opts="--transition-scale=1.0 --acoustic-scale=0.1 --self-loop-scale=0.1"

numiters=25   # Total number of iterations

realign_iters="5 10 15";
silphonelist=`cat data/silphones.csl`
numsubstates=1500 # Initial #-substates.
totsubstates=5000 # Target #-substates.
maxiterinc=15 # Last iter to increase #substates on.
incsubstates=$[($totsubstates-$numsubstates)/$maxiterinc] # per-iter increment for #substates
gselect_opt="--gselect=ark:gunzip -c $dir/gselect.gz|"
randprune=0.1
mkdir -p $dir

feats="ark:add-deltas --print-args=false scp:data/train.scp ark:- |"

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
         sgmm-align-compiled $scale_opts "$gselect_opt" --beam=8 --retry-beam=40 $dir/$iter.mdl \
           	"$srcgraphs" "$feats" \
         	ark:$dir/cur.ali 2> $dir/align.$iter.log || exit 1;
   fi
   if [ $iter -gt 0 ]; then
     flags=vMwcS
   else
     flags=vwcS
   fi
   if [ ! -f $dir/$[$iter+1].mdl ]; then
     sgmm-acc-stats-ali --update-flags=$flags "$gselect_opt" --rand-prune=$randprune --binary=false $dir/$iter.mdl "$feats" ark:$dir/cur.ali $dir/$iter.acc 2> $dir/acc.$iter.log  || exit 1;
     sgmm-est --update-flags=$flags --split-substates=$numsubstates --write-occs=$dir/$[$iter+1].occs $dir/$iter.mdl $dir/$iter.acc $dir/$[$iter+1].mdl 2> $dir/update.$iter.log || exit 1;
   fi
#  	rm $dir/$iter.mdl $dir/$iter.acc
#  	rm $dir/$iter.occs 
    if [ $iter -lt $maxiterinc ]; then
       numsubstates=$[$numsubstates+$incsubstates]
    fi
    iter=$[$iter+1];
done

( cd $dir; rm final.mdl final.occs 2>/dev/null; ln -s $iter.mdl final.mdl; ln -s $iter.occs final.occs )
