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


# 2a2 is as 2a but using smaller UBM (400 Guass), from ubm2b

# sgmm2a is speaker-independent SGMM building, starting from the system in tri1.
# The data subset (3500 utterances from si-84) is the same as in tri1, which
# means we can re-use its graphs.  


if [ -f path.sh ]; then . path.sh; fi

dir=exp/sgmm2a2
srcdir=exp/tri1
ubm=exp/ubm2b/final.ubm
srcmodel=$srcdir/final.mdl
scale_opts="--transition-scale=1.0 --acoustic-scale=0.1 --self-loop-scale=0.1"

numiters=35 # Total number of iterations.
realign_iters="5 15 25"; # realign a bit earlier than we did in tri2a, 
    # since SGMM system quite different
    # from normal triphone system.
maxiterinc=20 # By this iter, we have all the substates.
numleaves=3000 # was 2k for GMM system: incresaing it for SGMM system.
numsubstates=3000 # initial #-substates
totsubstates=8000 # a little less than #Gauss for baseline GMM system (10k)
incsubstates=$[($totsubstates-$numsubstates)/$maxiterinc] # per-iter increment for #substates

silphonelist=`cat data/silphones.csl`
randprune=0.1

mkdir -p $dir
cp $srcdir/train.scp $dir
cp $srcdir/train.tra $dir

# Do the expensive, parallelizable stuff on 3 cpus.
scripts/split_scp.pl $dir/train.scp  $dir/train{1,2,3}.scp
scripts/split_scp.pl $dir/train.tra  $dir/train{1,2,3}.tra

# also see featspart below, used for sub-parts of the features;
# try to keep them in sync.
feats="ark:add-deltas --print-args=false scp:$dir/train.scp ark:- |"
for n in 1 2 3; do
   featspart[$n]="ark:add-deltas --print-args=false scp:$dir/train${n}.scp ark:- |"
done

if [ ! -f $ubm ]; then
  echo "No UBM in $ubm";
  exit 1
fi

cp $srcdir/topo $dir

# Align all training data using old model (and old graphs, since we
# use the same data-subset as last time). 

echo "Aligning all training data"

rm -f $dir/.error
for n in 1 2 3; do
   gmm-align-compiled $scale_opts --beam=8 --retry-beam=40 $srcmodel \
       "ark:gunzip -c $srcdir/graphs${n}.fsts.gz|" "${featspart[$n]}" \
       "ark:|gzip -c >$dir/0.${n}.ali.gz" \
           2> $dir/align.0.${n}.log || touch $dir/.error &
done
wait;
[ -f $dir/.error ] &&  echo align error RE old system && exit 1


acc-tree-stats  --ci-phones=$silphonelist $srcmodel "$feats" \
  "ark:gunzip -c $dir/0.?.ali.gz|" $dir/treeacc 2> $dir/acc.tree.log  || exit 1;


# The next few commands are involved with making the questions
# for tree clustering.  The extra complexity vs. the RM recipe has
# to do with the desire to ask questions about the "real" phones
# ignoring things like stress and position-in-word, and ask questions
# separately about stress and position-in-word.

# Don't include silences as things to be clustered -> --nosil option.
scripts/make_shared_phones.sh --nosil | scripts/sym2int.pl data/phones.txt > $dir/phone_sets.list
cluster-phones $dir/treeacc $dir/phone_sets.list $dir/questions.txt 2> $dir/cluster_phones.log || exit 1;
scripts/int2sym.pl data/phones.txt < $dir/questions.txt > $dir/questions_syms.txt
scripts/make_extra_questions.sh | cat $dir/questions_syms.txt - > $dir/questions_syms_all.txt
scripts/sym2int.pl data/phones.txt < $dir/questions_syms_all.txt > $dir/questions_all.txt

compile-questions $dir/topo $dir/questions_all.txt $dir/questions.qst 2>$dir/compile_questions.log || exit 1;

scripts/make_roots.sh > $dir/roots_syms.txt
scripts/sym2int.pl --ignore-oov data/phones.txt  < $dir/roots_syms.txt > $dir/roots.txt

build-tree --verbose=1 --max-leaves=$numleaves \
  $dir/treeacc $dir/roots.txt \
  $dir/questions.qst $dir/topo $dir/tree  2> $dir/train_tree.log || exit 1;

# the sgmm-init program accepts a GMM, so we just create a temporary GMM "0.gmm"

gmm-init-model  --write-occs=$dir/0.occs  \
    $dir/tree $dir/treeacc $dir/topo $dir/0.gmm 2> $dir/init_gmm.log || exit 1;

sgmm-init $dir/0.gmm $ubm $dir/0.mdl 2> $dir/init_sgmm.log || exit 1;

rm $dir/0.gmm

rm $dir/treeacc

for n in 1 2 3; do
  sgmm-gselect $dir/0.mdl "${featspart[$n]}" ark,t:- 2>$dir/gselect$n.log | \
   gzip -c > $dir/gselect${n}.gz || touch $dir/.error &
done
wait
[ -f $dir/.error ] && echo "Error in gselect phase" && exit 1;


# Convert alignments generated from previous model, to use as 
# initial alignments.

for n in 1 2 3; do
  convert-ali $srcmodel $dir/0.mdl $dir/tree \
      "ark:gunzip -c $dir/0.$n.ali.gz|" \
      "ark:|gzip -c > $dir/cur$n.ali.gz" \
     2>$dir/convert.$n.log || exit 1; # don't parallelize: mostly I/O.
done
rm $dir/0.?.ali.gz

# Make training graphs
echo "Compiling training graphs"

rm -f $dir/.error
for n in 1 2 3; do
  compile-train-graphs $dir/tree $dir/0.mdl  data/L.fst ark:$dir/train${n}.tra \
     "ark:|gzip -c > $dir/graphs${n}.fsts.gz" \
     2>$dir/compile_graphs.${n}.log || touch $dir/.error &
done
wait
[ -f $dir/.error ] &&  echo compile-graphs error && exit 1

x=0
while [ $x -lt $numiters ]; do
   echo "Pass $x"
   if echo $realign_iters | grep -w $x >/dev/null; then
     echo "Aligning data"
     rm -f $dir/.error
     for n in 1 2 3; do
       sgmm-align-compiled "--gselect=ark:gunzip -c $dir/gselect$n.gz|" \
           $scale_opts --beam=8 --retry-beam=40 $dir/$x.mdl \
           "ark:gunzip -c $dir/graphs${n}.fsts.gz|" "${featspart[$n]}" \
           "ark:|gzip -c >$dir/cur${n}.ali.gz" 2> $dir/align.$x.$n.log \
             || touch $dir/.error &
     done
     wait 
     [ -f $dir/.error ] && echo error aligning data && exit 1
   fi
   if [ $x -gt 0 ]; then
     flags=vMwcS
   else
     flags=vwcS
   fi
   for n in 1 2 3; do
     sgmm-acc-stats-ali --update-flags=$flags "--gselect=ark:gunzip -c $dir/gselect$n.gz|" \
       --rand-prune=$randprune --binary=true $dir/$x.mdl "${featspart[$n]}" \
      "ark:gunzip -c $dir/cur$n.ali.gz|" $dir/$x.$n.acc 2> $dir/acc.$x.$n.log \
        || touch $dir/.error &
   done
   wait;
   [ -f $dir/.error ] && echo error accumulating stats on iter $x && exit 1  
   sgmm-est --update-flags=$flags --split-substates=$numsubstates --write-occs=$dir/$[$x+1].occs \
       $dir/$x.mdl "sgmm-sum-accs - $dir/$x.?.acc|" $dir/$[$x+1].mdl  2> $dir/update.$x.log || exit 1;
   rm $dir/$x.mdl $dir/$x.?.acc $dir/$x.occs 2>/dev/null
   if [ $x -lt $maxiterinc ]; then 
     numsubstates=$[$numsubstates+$incsubstates]
   fi
   x=$[$x+1];
done

( cd $dir; rm final.mdl final.occs 2>/dev/null; ln -s $x.mdl final.mdl; ln -s $x.occs final.occs )

