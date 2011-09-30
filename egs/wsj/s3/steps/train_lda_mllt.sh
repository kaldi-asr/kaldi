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

if [ $# != 6 ]; then
   echo "Usage: steps/train_lda_mllt.sh <num-leaves> <tot-gauss> <data-dir> <lang-dir> <ali-dir> <exp-dir>"
   echo " e.g.: steps/train_lda_mllt.sh 2500 15000 data/train_si84 data/lang exp/tri1_ali_si84 exp/tri2b"
   exit 1;
fi

if [ -f path.sh ]; then . path.sh; fi

numleaves=$1
totgauss=$2
data=$3
lang=$4
alidir=$5
dir=$6

if [ ! -f $alidir/final.mdl -o ! -f $alidir/0.ali.gz -o ! -f $alidir/3.ali.gz ]; then
  echo "Error: alignment dir $alidir does not contain final.mdl and {0,1,2,3}.ali.gz"
  exit 1;
fi

scale_opts="--transition-scale=1.0 --acoustic-scale=0.1 --self-loop-scale=0.1"
realign_iters="10 20 30";
mllt_iters="2 4 6 12";
oov_sym="<SPOKEN_NOISE>" # Map OOVs to this in training.
grep SPOKEN_NOISE $lang/words.txt >/dev/null || echo "Warning: SPOKEN_NOISE not in dictionary"
silphonelist=`cat $lang/silphones.csl`
numiters=35    # Number of iterations of training
maxiterinc=25 # Last iter to increase #Gauss on.
numgauss=$numleaves
incgauss=$[($totgauss-$numgauss)/$maxiterinc] # per-iter increment for #Gauss
randprune=4.0 # This is approximately the ratio by which we will speed up the
              # LDA and MLLT calculations via randomized pruning.

mkdir -p $dir/log

if [ ! -f $data/split4 -o $data/split4 -ot $data/feats.scp ]; then
  scripts/split_data.sh $data 4
fi

# feats0 is all the feats, transformed with 0.mat-- just needed for tree accumulation.
feats0="ark:apply-cmvn --norm-vars=false --utt2spk=ark:$data/utt2spk \"ark:cat $alidir/*.cmvn|\" scp:$data/feats.scp ark:- | splice-feats ark:- ark:- | transform-feats $dir/0.mat ark:- ark:- |"

# featspart[n] gets overwritten later in the script.
for n in 0 1 2 3; do
  splicedfeatspart[$n]="ark:apply-cmvn --norm-vars=false --utt2spk=ark:$data/utt2spk ark:$alidir/$n.cmvn scp:$data/split4/$n/feats.scp ark:- | splice-feats ark:- ark:- |"
  featspart[$n]="${splicedfeatspart[$n]} transform-feats $dir/0.mat ark:- ark:- |"
done


echo "Accumulating LDA statistics."

rm $dir/.error 2>/dev/null

for n in 0 1 2 3; do
( ali-to-post "ark:gunzip -c $alidir/$n.ali.gz|" ark:- | \
   weight-silence-post 0.0 $silphonelist $alidir/final.mdl ark:- ark:- | \
   acc-lda --rand-prune=$randprune $alidir/final.mdl "${splicedfeatspart[$n]}" ark,s,cs:- \
     $dir/lda.$n.acc ) 2>$dir/log/lda_acc.$n.log || touch $dir/.error &
done
wait
[ -f $dir/.error ] && echo "Error accumulating LDA stats" && exit 1;
est-lda $dir/0.mat $dir/lda.*.acc 2>$dir/log/lda_est.log || exit 1; # defaults to dim=40
rm $dir/lda.*.acc
cur_lda=$dir/0.mat

# The next stage assumes we won't need the context of silence, which
# assumes something about $lang/roots.txt, but it seems pretty safe.
echo "Accumulating tree stats"
acc-tree-stats  --ci-phones=$silphonelist $alidir/final.mdl "$feats0" \
  "ark:gunzip -c $alidir/?.ali.gz|" $dir/treeacc 2> $dir/log/acc_tree.log  || exit 1;

echo "Computing questions for tree clustering"
# preparing questions, roots file...
scripts/sym2int.pl $lang/phones.txt $lang/phonesets_cluster.txt > $dir/phonesets.txt || exit 1;
cluster-phones $dir/treeacc $dir/phonesets.txt $dir/questions.txt 2> $dir/log/questions.log || exit 1;
scripts/sym2int.pl $lang/phones.txt $lang/extra_questions.txt >> $dir/questions.txt
compile-questions $lang/topo $dir/questions.txt $dir/questions.qst 2>$dir/log/compile_questions.log || exit 1;
scripts/sym2int.pl --ignore-oov $lang/phones.txt $lang/roots.txt > $dir/roots.txt

echo "Building tree"
build-tree --verbose=1 --max-leaves=$numleaves \
    $dir/treeacc $dir/roots.txt \
    $dir/questions.qst $lang/topo $dir/tree  2> $dir/log/train_tree.log || exit 1;

gmm-init-model  --write-occs=$dir/1.occs  \
    $dir/tree $dir/treeacc $lang/topo $dir/1.mdl 2> $dir/log/init_model.log || exit 1;

gmm-mixup --mix-up=$numgauss $dir/1.mdl $dir/1.occs $dir/1.mdl \
   2>$dir/log/mixup.log || exit 1;

rm $dir/treeacc

# Convert alignments in $alidir, to use as initial alignments.
# This assumes that $alidir was split in 4 pieces, just like the
# current dir.

echo "Converting old alignments"
for n in 0 1 2 3; do
  convert-ali $alidir/final.mdl $dir/1.mdl $dir/tree \
   "ark:gunzip -c $alidir/$n.ali.gz|" "ark:|gzip -c >$dir/$n.ali.gz" \
    2>$dir/log/convert$n.log  || exit 1;
done

# Make training graphs (this is split in 4 parts).
echo "Compiling training graphs"
rm $dir/.error 2>/dev/null
for n in 0 1 2 3; do
  compile-train-graphs $dir/tree $dir/1.mdl  $lang/L.fst  \
    "ark:scripts/sym2int.pl --map-oov \"$oov_sym\" --ignore-first-field $lang/words.txt < $data/split4/$n/text |" \
    "ark:|gzip -c >$dir/$n.fsts.gz"  2>$dir/log/compile_graphs$n.log  || touch $dir/.error&
done
wait;
[ -f $dir/.error ] && echo "Error compiling training graphs" && exit 1;

x=1
while [ $x -lt $numiters ]; do
   echo Pass $x
   if echo $realign_iters | grep -w $x >/dev/null; then
     echo "Aligning data"
     for n in 0 1 2 3; do
       gmm-align-compiled $scale_opts --beam=10 --retry-beam=40 $dir/$x.mdl \
              "ark:gunzip -c $dir/$n.fsts.gz|" "${featspart[$n]}" \
              "ark:|gzip -c >$dir/$n.ali.gz" 2> $dir/log/align.$x.$n.log || touch $dir/.error &
     done
     wait;
     [ -f $dir/.error ] && echo "Error aligning data on iteration $x" && exit 1;
   fi
   if echo $mllt_iters | grep -w $x >/dev/null; then
     echo "Estimating MLLT"
     for n in 0 1 2 3; do
      ( ali-to-post "ark:gunzip -c $dir/$n.ali.gz|" ark:- | \
         weight-silence-post 0.0 $silphonelist $dir/$x.mdl ark:- ark:- | \
         gmm-acc-mllt --rand-prune=$randprune --binary=false $dir/$x.mdl \
           "${featspart[$n]}" ark:- $dir/$x.$n.macc ) 2> $dir/log/macc.$x.$n.log || touch $dir/.error &
       featspart[$n]="${splicedfeatspart[$n]} transform-feats $dir/$x.mat ark:- ark:- |"       
     done
     wait
     [ -f $dir/.error ] && echo "Error accumulating MLLT stats on iter $x" && exit 1;
     est-mllt $dir/$x.mat.new $dir/$x.{0,1,2,3}.macc 2> $dir/log/mupdate.$x.log || exit 1;
     gmm-transform-means --binary=false $dir/$x.mat.new $dir/$x.mdl $dir/$x.mdl \
       2> $dir/log/transform_means.$x.log || exit 1;
     compose-transforms --print-args=false $dir/$x.mat.new $cur_lda $dir/$x.mat || exit 1;
     cur_lda=$dir/$x.mat
     rm $dir/$x.*.macc
   fi
   for n in 0 1 2 3; do
     gmm-acc-stats-ali --binary=false $dir/$x.mdl "${featspart[$n]}" \
       "ark:gunzip -c $dir/$n.ali.gz|" $dir/$x.$n.acc \
      2>$dir/log/acc.$x.$n.log  || touch $dir/.error &
   done
   wait;
   [ -f $dir/.error ] && echo "Error accumulating stats on iteration $x" && exit 1;
   gmm-est --write-occs=$dir/$[$x+1].occs --mix-up=$numgauss $dir/$x.mdl \
      "gmm-sum-accs - $dir/$x.{0,1,2,3}.acc |" $dir/$[$x+1].mdl 2> $dir/log/update.$x.log || exit 1;
   rm $dir/$x.mdl $dir/$x.{0,1,2,3}.acc
   rm $dir/$x.occs 
   if [[ $x -le $maxiterinc ]]; then 
      numgauss=$[$numgauss+$incgauss];
   fi
   x=$[$x+1];
done

( cd $dir; rm final.mdl 2>/dev/null; ln -s $x.mdl final.mdl; ln -s $x.occs final.occs;
  ln -s `basename $cur_lda` final.mat )

echo Done
