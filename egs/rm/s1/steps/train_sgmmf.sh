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
# You must run train_ubma.sh first, as well as train_tri1.sh
# We rely on the UBM exp/ubm/4.ubm being there

if [ -f path.sh ]; then . path.sh; fi

dir=exp/sgmmf
ubm=exp/ubma/4.ubm
srcdir=exp/tri1
srcmodel=$srcdir/final.mdl
srcgraphs="ark:gunzip -c $srcdir/graphs.fsts.gz|"
scale_opts="--transition-scale=1.0 --acoustic-scale=0.1 --self-loop-scale=0.1"

numiters=25   # Total number of iterations

realign_iters="5 10 15";
silphonelist=`cat data/silphones.csl`
numleaves=2500
numsubstates=2500 # Initial #-substates.
totsubstates=7500 # Target #-substates.
maxiterinc=15 # Last iter to increase #substates on.

incsubstates=$[($totsubstates-$numsubstates)/$maxiterinc] # per-iter increment for #substates
gselect_opt="--gselect=ark,s,cs:gunzip -c $dir/gselect.gz|"
randprune=0.1
mkdir -p $dir

feats="ark,s,cs:add-deltas --print-args=false scp:data/train.scp ark:- |"

cp $srcdir/topo $dir

if [ ! -f $ubm ]; then
  echo "No UBM in $ubm";
  exit 1
fi

# Get the ilabel_info (ilabels) because we need a complete list of
# possible triphones and this is just the most convenient way to get it.
grep '#' data/phones_disambig.txt | awk '{print $2}' > $dir/disambig_phones.list
fstdeterminize data/L_disambig.fst | fstrmsymbols $dir/disambig_phones.list | \
  fstcomposecontext $dir/ilabels >/dev/null

tail -n +2 $dir/ilabels | awk '{print $2, $3, $4;}' > $dir/triphones

for x in `echo $silphonelist | sed 's/://'`; do
  cat $dir/triphones | awk '{if ($2 == '$x') { print 0, $2, 0; } else { print; }} ' | \
    sort | uniq > $dir/tmp; 
  mv $dir/tmp $dir/triphones; 
done

# OK, now we have our triphone list.



echo "aligning all training data"
if [ ! -f $dir/0.ali ]; then
  gmm-align-compiled  $scale_opts --beam=8 --retry-beam=40  $srcmodel "$srcgraphs" \
        "$feats" ark,t:$dir/0.ali 2> $dir/align.0.log || exit 1;
fi

# Initialize tree with untied triphones.
init-tree-special $dir/triphones $dir/topo $dir/states $dir/tree 2>$dir/init_tree.log


sgmm-init $dir/topo $dir/tree $ubm $dir/0.mdl 2> $dir/init_sgmm.log || exit 1;

rm $dir/0.gmm

if [ ! -f $dir/gselect.gz ]; then
 sgmm-gselect $dir/0.mdl "$feats" ark,t:- 2>$dir/gselect.log | gzip -c > $dir/gselect.gz || exit 1;
fi

convert-ali  $srcmodel $dir/0.mdl $dir/tree ark:$dir/0.ali \
  ark:$dir/cur.ali 2>$dir/convert.log 

rm $dir/0.ali

# Make training graphs
echo "Compiling training graphs"
compile-train-graphs $dir/tree $dir/0.mdl  data/L.fst ark:data/train.tra \
   "ark:|gzip -c >$dir/graphs.fsts.gz"  2>$dir/compile_graphs.log  || exit 1 

iter=0
while [ $iter -lt $numiters ]; do
   echo "Pass $iter ... "
   if echo $realign_iters | grep -w $iter >/dev/null; then
      echo "Aligning data"
      sgmm-align-compiled $spkvecs_opt $scale_opts "$gselect_opt" --beam=8 \
          --retry-beam=40 $dir/$iter.mdl "ark:gunzip -c $dir/graphs.fsts.gz|" "$feats" \
      	ark:$dir/cur.ali 2> $dir/align.$iter.log || exit 1;
   fi
   if [ $iter -gt 0 ]; then
     flags=vMwcS
   else
     flags=vwcS
   fi
   sgmm-acc-stats-ali --update-flags=$flags "$gselect_opt" --rand-prune=$randprune --binary=false $dir/$iter.mdl "$feats" ark:$dir/cur.ali $dir/$iter.acc 2> $dir/acc.$iter.log  || exit 1;
   sgmm-est --update-flags=$flags --split-substates=$numsubstates --write-occs=$dir/$[$iter+1].occs $dir/$iter.mdl $dir/$iter.acc $dir/$[$iter+1].mdl 2> $dir/update.$iter.log || exit 1;

   rm $dir/$iter.mdl $dir/$iter.acc $dir/$iter.occs 
   if [ $iter -lt $maxiterinc ]; then
     numsubstates=$[$numsubstates+$incsubstates]
   fi
   iter=$[$iter+1];
done

( cd $dir; rm final.mdl final.occs 2>/dev/null; ln -s $iter.mdl final.mdl; ln -s $iter.occs final.occs )
