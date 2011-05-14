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

# This (train_tri2c) is training with mean normalization (you could
# modify options in this script to do variance normalization).

if [ -f path.sh ]; then . path.sh; fi
dir=exp/tri2c
srcdir=exp/tri1
srcmodel=$srcdir/final.mdl
srcgraphs="ark:gunzip -c $srcdir/graphs.fsts.gz|"
scale_opts="--transition-scale=1.0 --acoustic-scale=0.1 --self-loop-scale=0.1"
mkdir -p $dir
cp $srcdir/topo $dir
norm_vars=false
after_deltas=false
per_spk=true

numiters=30    # Number of iterations of training
maxiterinc=20 # Last iter to increase #Gauss on.
numiters_et=15 # Before this, update et.
numleaves=1500
numgauss=$numleaves
totgauss=7000 # Target #Gaussians
incgauss=$[($totgauss-$numgauss)/$maxiterinc] # per-iter increment for #Gauss
realign_iters="10 15 20 25";

silphonelist=`cat data/silphones.csl`

if [ $per_spk == "true" ]; then
  spk2utt_opt=--spk2utt=ark:data/train.spk2utt
  utt2spk_opt=--utt2spk=ark:data/train.utt2spk
else
  spk2utt_opt=
  utt2spk_opt=
fi

srcfeats="ark:add-deltas --print-args=false scp:data/train.scp ark:- |"

echo "Computing cepstral mean and variance stats."
# compute mean and variance stats.
if [ $after_deltas == true ]; then
  compute-cmvn-stats $spk2utt_opt "$srcfeats" ark:$dir/cmvn.ark 2>$dir/cmvn.log
  feats="ark:add-deltas --print-args=false scp:data/train.scp ark:- | apply-cmvn --norm-vars=$norm_vars $utt2spk_opt ark:$dir/cmvn.ark ark:- ark:- |"
else 
  compute-cmvn-stats --spk2utt=ark:data/train.spk2utt scp:data/train.scp \
      ark:$dir/cmvn.ark 2>$dir/cmvn.log
  feats="ark:apply-cmvn --norm-vars=$norm_vars $utt2spk_opt ark:$dir/cmvn.ark scp:data/train.scp ark:- | add-deltas --print-args=false ark:- ark:- |"
fi


echo "aligning all training data"
gmm-align-compiled  $scale_opts --beam=8 --retry-beam=40  $srcmodel \
  "$srcgraphs" "$srcfeats" ark,t:$dir/0.ali 2> $dir/align.0.log || exit 1;

acc-tree-stats  --ci-phones=$silphonelist $srcmodel "$feats" ark:$dir/0.ali $dir/treeacc 2> $dir/acc.tree.log  || exit 1;


cat data/phones.txt | awk '{print $NF}' | grep -v -w 0 > $dir/phones.list
cluster-phones $dir/treeacc $dir/phones.list $dir/questions.txt 2> $dir/questions.log || exit 1;
scripts/int2sym.pl data/phones.txt < $dir/questions.txt > $dir/questions_syms.txt
compile-questions $dir/topo $dir/questions.txt $dir/questions.qst 2>$dir/compile_questions.log || exit 1;

scripts/make_roots.pl --separate data/phones.txt `cat data/silphones.csl` shared split > $dir/roots.txt 2>$dir/roots.log || exit 1;

build-tree --verbose=1 --max-leaves=$numleaves \
    $dir/treeacc $dir/roots.txt \
    $dir/questions.qst $dir/topo $dir/tree  2> $dir/train_tree.log || exit 1;

gmm-init-model  --write-occs=$dir/1.occs  \
    $dir/tree $dir/treeacc $dir/topo $dir/1.mdl 2> $dir/init_model.log || exit 1;


# Convert alignments generated from previous model, to use as initial alignments.

rm $dir/treeacc

convert-ali  $srcmodel $dir/1.mdl $dir/tree ark:$dir/0.ali ark:$dir/cur.ali 2>$dir/convert.log 
  # Debug step only: convert back and check they're the same.
  convert-ali $dir/1.mdl $srcmodel $srcdir/tree ark:$dir/cur.ali ark,t:- \
   2>/dev/null | cmp - $dir/0.ali || exit 1; 

rm $dir/0.ali

# Make training graphs
echo "Compiling training graphs"
compile-train-graphs $dir/tree $dir/1.mdl  data/L.fst ark:data/train.tra ark:$dir/graphs.fsts \
    2>$dir/compile_graphs.log || exit 1 

x=1
while [ $x -lt $numiters ]; do
   echo pass $x
   if echo $realign_iters | grep -w $x >/dev/null; then
     echo "Aligning data"
     gmm-align-compiled $scale_opts --beam=8 --retry-beam=40 $dir/$x.mdl ark:$dir/graphs.fsts "$feats" \
             ark:$dir/cur.ali 2> $dir/align.$x.log || exit 1;
   fi
   gmm-acc-stats-ali --binary=false $dir/$x.mdl "$feats" ark:$dir/cur.ali $dir/$x.acc 2> $dir/acc.$x.log  || exit 1;
   gmm-est --mix-up=$numgauss $dir/$x.mdl $dir/$x.acc $dir/$[$x+1].mdl 2> $dir/update.$x.log || exit 1;
   rm $dir/$x.mdl $dir/$x.acc
   if [ $x -le $maxiterinc ]; then
      numgauss=$[$numgauss+$incgauss];
   fi
   x=$[$x+1];
done

( cd $dir; rm final.mdl 2>/dev/null; ln -s $x.mdl final.mdl )
