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
# Triphone model training, using delta-delta features and cepstral
# mean normalization.  It starts from an existing directory (e.g.
# exp/mono), supplied as an argument, which is assumed to be built using
# the same type of features.

nj=4
cmd=scripts/run.pl
for x in 1 2; do
  if [ $1 == "--num-jobs" ]; then
     shift
     nj=$1
     shift
  fi
  if [ $1 == "--cmd" ]; then
     shift
     cmd=$1
     shift
  fi  
done

if [ $# != 6 ]; then
   echo "Usage: steps/train_deltas.sh <num-leaves> <tot-gauss> <data-dir> <lang-dir> <ali-dir> <exp-dir>"
   echo " e.g.: steps/train_deltas.sh 2000 10000 data/train_si84_half data/lang exp/mono_ali exp/tri1"
   exit 1;
fi

if [ -f path.sh ]; then . path.sh; fi

numleaves=$1
totgauss=$2
data=$3
lang=$4
alidir=$5
dir=$6

if [ ! -f $alidir/final.mdl ]; then
  echo "Error: alignment dir $alidir does not contain final.mdl"
  exit 1;
fi

scale_opts="--transition-scale=1.0 --acoustic-scale=0.1 --self-loop-scale=0.1"
realign_iters="10 20 30";
oov_sym="<SPOKEN_NOISE>" # Map OOVs to this in training.
grep SPOKEN_NOISE $lang/words.txt >/dev/null || echo "Warning: SPOKEN_NOISE not in dictionary"
silphonelist=`cat $lang/silphones.csl`
numiters=35    # Number of iterations of training
maxiterinc=25 # Last iter to increase #Gauss on.
numgauss=$numleaves
incgauss=$[($totgauss-$numgauss)/$maxiterinc] # per-iter increment for #Gauss


mkdir -p $dir/log

if [ ! -f $data/split$nj -o $data/split$nj -ot $data/feats.scp ]; then
  scripts/split_data.sh $data $nj
fi

feats="ark:apply-cmvn --norm-vars=false --utt2spk=ark:$data/utt2spk \"ark:cat $alidir/*.cmvn|\" scp:$data/feats.scp ark:- | add-deltas ark:- ark:- |"
for n in `get_splits.pl $nj`; do
  featspart[$n]="ark:apply-cmvn --norm-vars=false --utt2spk=ark:$data/split$nj/$n/utt2spk ark:$alidir/$n.cmvn scp:$data/split$nj/$n/feats.scp ark:- | add-deltas ark:- ark:- |"
done


# The next stage assumes we won't need the context of silence, which
# assumes something about $lang/roots.txt, but it seems pretty safe.
echo "Accumulating tree stats"
$cmd $dir/log/acc_tree.log \
  acc-tree-stats  --ci-phones=$silphonelist $alidir/final.mdl "$feats" \
    "ark:gunzip -c $alidir/*.ali.gz|" $dir/treeacc || exit 1;

echo "Computing questions for tree clustering"
# preparing questions, roots file...
scripts/sym2int.pl $lang/phones.txt $lang/phonesets_cluster.txt > $dir/phonesets.txt || exit 1;
cluster-phones $dir/treeacc $dir/phonesets.txt $dir/questions.txt 2> $dir/log/questions.log || exit 1;
scripts/sym2int.pl $lang/phones.txt $lang/extra_questions.txt >> $dir/questions.txt
compile-questions $lang/topo $dir/questions.txt $dir/questions.qst 2>$dir/log/compile_questions.log || exit 1;
scripts/sym2int.pl --ignore-oov $lang/phones.txt $lang/roots.txt > $dir/roots.txt

echo "Building tree"
$cmd $dir/log/train_tree.log \
  build-tree --verbose=1 --max-leaves=$numleaves \
    $dir/treeacc $dir/roots.txt \
    $dir/questions.qst $lang/topo $dir/tree || exit 1;

gmm-init-model  --write-occs=$dir/1.occs  \
    $dir/tree $dir/treeacc $lang/topo $dir/1.mdl 2> $dir/log/init_model.log || exit 1;

gmm-mixup --mix-up=$numgauss $dir/1.mdl $dir/1.occs $dir/1.mdl \
   2>$dir/log/mixup.log || exit 1;

rm $dir/treeacc

# Convert alignments in $alidir, to use as initial alignments.
# This assumes that $alidir was split in $nj pieces, just like the
# current dir.  Just do this locally-- it's very fast.

echo "Converting old alignments"
for n in `get_splits.pl $nj`; do
  convert-ali $alidir/final.mdl $dir/1.mdl $dir/tree \
   "ark:gunzip -c $alidir/$n.ali.gz|" "ark:|gzip -c >$dir/$n.ali.gz" \
    2>$dir/log/convert$n.log  || exit 1;
done

# Make training graphs (this is split in $nj parts).
echo "Compiling training graphs"
rm $dir/.error 2>/dev/null
for n in `get_splits.pl $nj`; do
  $cmd $dir/log/compile_graphs$n.log \
    compile-train-graphs $dir/tree $dir/1.mdl  $lang/L.fst  \
      "ark:scripts/sym2int.pl --map-oov \"$oov_sym\" --ignore-first-field $lang/words.txt < $data/split$nj/$n/text |" \
      "ark:|gzip -c >$dir/$n.fsts.gz" || touch $dir/.error &
done
wait;
[ -f $dir/.error ] && echo "Error compiling training graphs" && exit 1;

x=1
while [ $x -lt $numiters ]; do
   echo Pass $x
   if echo $realign_iters | grep -w $x >/dev/null; then
     echo "Aligning data"
     for n in `get_splits.pl $nj`; do
       $cmd $dir/log/align.$x.$n.log \
         gmm-align-compiled $scale_opts --beam=10 --retry-beam=40 $dir/$x.mdl \
           "ark:gunzip -c $dir/$n.fsts.gz|" "${featspart[$n]}" \
           "ark:|gzip -c >$dir/$n.ali.gz" || touch $dir/.error &
     done
     wait;
     [ -f $dir/.error ] && echo "Error aligning data on iteration $x" && exit 1;
   fi
   for n in `get_splits.pl $nj`; do
     $cmd $dir/log/acc.$x.$n.log
     gmm-acc-stats-ali --binary=false $dir/$x.mdl "${featspart[$n]}" \
       "ark:gunzip -c $dir/$n.ali.gz|" $dir/$x.$n.acc || touch $dir/.error &
   done
   wait;
   [ -f $dir/.error ] && echo "Error accumulating stats on iteration $x" && exit 1;
   $cmd $dir/log/update.$x.log \
     gmm-est --write-occs=$dir/$[$x+1].occs --mix-up=$numgauss $dir/$x.mdl \
       "gmm-sum-accs - $dir/$x.*.acc |" $dir/$[$x+1].mdl || exit 1;
   rm $dir/$x.mdl $dir/$x.*.acc
   rm $dir/$x.occs 
   if [[ $x -le $maxiterinc ]]; then 
      numgauss=$[$numgauss+$incgauss];
   fi
   x=$[$x+1];
done

( cd $dir; rm final.mdl 2>/dev/null; ln -s $x.mdl final.mdl; ln -s $x.occs final.occs )

# Print out summary of the warning messages.
for x in $dir/log/*.log; do 
  n=`grep WARNING $x | wc -l`; 
  if [ $n -ne 0 ]; then echo $n warnings in $x; fi; 
done

echo Done
