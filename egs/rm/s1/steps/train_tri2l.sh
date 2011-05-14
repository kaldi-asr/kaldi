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

# This (train_tri2l) is training with splice-9-frames+LDA features,
# plus MLLT plus CMLLR/fMLLR (i.e. speaker adapted training).

if [ -f path.sh ]; then . path.sh; fi
dir=exp/tri2l
srcdir=exp/tri1
srcmodel=$srcdir/final.mdl
srcgraphs="ark:gunzip -c $srcdir/graphs.fsts.gz|"
scale_opts="--transition-scale=1.0 --acoustic-scale=0.1 --self-loop-scale=0.1"
numiters=30    # Number of iterations of training
maxiterinc=20 # Last iter to increase #Gauss on.
numleaves=1500
numgauss=$numleaves
totgauss=7000 # Target #Gaussians
incgauss=$[($totgauss-$numgauss)/$maxiterinc] # per-iter increment for #Gauss
silphonelist=`cat data/silphones.csl`
realign_iters="10 15 20 25";  
mllt_iters="2 4 6 8";
fmllr_iters="9 14 19"
spk2utt_opt="--spk2utt=ark:data/train.spk2utt"
utt2spk_opt="--utt2spk=ark:data/train.utt2spk"

# feats corresponding to original model
srcfeats="ark:add-deltas --print-args=false scp:data/train.scp ark:- |"
feats="ark:splice-feats scp:data/train.scp ark:- | transform-feats $dir/0.mat ark:- ark:-|"
# Subset of features used to train LDA and MLLT transforms.
featsub="ark:scripts/subset_scp.pl 800 data/train.scp | splice-feats scp:- ark:- | transform-feats $dir/0.mat ark:- ark:-|"

mkdir -p $dir
cp $srcdir/topo $dir

echo "aligning all training data"

gmm-align-compiled  $scale_opts --beam=8 --retry-beam=40  $srcmodel \
   "$srcgraphs" "$srcfeats" ark,t:$dir/0.ali 2> $dir/align.0.log || exit 1;

( ali-to-post ark:$dir/0.ali ark:- | \
   weight-silence-post 0.0 $silphonelist $srcmodel ark:- ark:- | \
   acc-lda $srcmodel "ark:scripts/subset_scp.pl 800 data/train.scp | splice-feats scp:- ark:- |" \
       ark:- $dir/lda.acc ) 2>$dir/lda_acc.log
est-lda $dir/0.mat $dir/lda.acc 2>$dir/lda_est.log


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

rm $dir/treeacc

# Convert alignments generated from monophone model, to use as initial alignments.

convert-ali $srcmodel $dir/1.mdl $dir/tree ark:$dir/0.ali ark:$dir/cur.ali 2>$dir/convert.log 
  # Debug step only: convert back and check they're the same.
  convert-ali $dir/1.mdl $srcmodel $srcdir/tree ark:$dir/cur.ali ark,t:- \
   2>/dev/null | cmp - $dir/0.ali || exit 1; 

rm $dir/0.ali

# Make training graphs
echo "Compiling training graphs"
compile-train-graphs $dir/tree $dir/1.mdl  data/L.fst ark:data/train.tra \
    "ark:|gzip -c >$dir/graphs.fsts.gz"  2>$dir/compile_graphs.log || exit 1 

cur_lda=$dir/0.mat
x=1
while [ $x -lt $numiters ]; do
   echo pass $x
   if echo $realign_iters | grep -w $x >/dev/null; then
     echo "Aligning data"
     gmm-align-compiled $scale_opts --beam=8 --retry-beam=40 $dir/$x.mdl \
            "ark:gunzip -c $dir/graphs.fsts.gz|" "$feats" \
             ark:$dir/cur.ali 2> $dir/align.$x.log || exit 1;
   fi
   if echo $fmllr_iters | grep -w $x >/dev/null; then # Compute CMLLR transforms.
     sifeats="ark:splice-feats scp:data/train.scp ark:- | transform-feats $cur_lda ark:- ark:-|"
    ( ali-to-post ark:$dir/cur.ali ark:- | \
      weight-silence-post 0.0 $silphonelist $dir/$x.mdl ark:- ark:- | \
      gmm-post-to-gpost $dir/$x.mdl "$feats" ark:- ark:- | \
      gmm-est-fmllr-gpost $spk2utt_opt $dir/$x.mdl "$sifeats" ark,s,cs:- ark:$dir/tmp.trans ) \
           2> $dir/trans.$x.log  || exit 1;
     mv $dir/tmp.trans $dir/cur.trans
     feats="ark:splice-feats scp:data/train.scp ark:- | transform-feats $cur_lda ark:- ark:- | transform-feats $utt2spk_opt ark:$dir/cur.trans ark:- ark:- |"
   fi
   if echo $mllt_iters | grep -w $x >/dev/null; then # Do MLLT update.
    ( ali-to-post ark:$dir/cur.ali ark:- | \
       weight-silence-post 0.0 $silphonelist $dir/$x.mdl ark:- ark:- | \
       gmm-acc-mllt --binary=false $dir/$x.mdl "$featsub" ark:- $dir/$x.macc ) 2> $dir/macc.$x.log  || exit 1;

     est-mllt $dir/$x.mat.new $dir/$x.macc 2> $dir/mupdate.$x.log || exit 1;
     gmm-transform-means --binary=false $dir/$x.mat.new $dir/$x.mdl $dir/$[$x+1].mdl 2> $dir/transform_means.$x.log || exit 1;
     compose-transforms --print-args=false $dir/$x.mat.new $cur_lda $dir/$x.mat || exit 1;
     cur_lda=$dir/$x.mat


     feats="ark:splice-feats scp:data/train.scp ark:- | transform-feats $cur_lda ark:- ark:-|"
     # Subset of features used to train MLLT transforms.
     featsub="ark:scripts/subset_scp.pl 800 data/train.scp | splice-feats scp:- ark:- | transform-feats $cur_lda ark:- ark:-|"
   else # do GMM update.
     gmm-acc-stats-ali --binary=false $dir/$x.mdl "$feats" ark:$dir/cur.ali $dir/$x.acc 2> $dir/acc.$x.log  || exit 1;
     gmm-est --mix-up=$numgauss $dir/$x.mdl $dir/$x.acc $dir/$[$x+1].mdl 2> $dir/update.$x.log || exit 1;
   fi
   rm $dir/$x.mdl $dir/$x.acc 2>/dev/null
   if [ $x -le $maxiterinc ]; then 
      numgauss=$[$numgauss+$incgauss];
   fi
   x=$[$x+1]
done

defaultfeats="ark:splice-feats scp:data/train.scp ark:- | transform-feats $cur_lda ark:- ark:-|"

# Accumulate stats for "alignment model" which is as the model but with
# the unadapted, default features (shares Gaussian-level alignments).
( ali-to-post ark:$dir/cur.ali ark:-  | \
  gmm-acc-stats-twofeats $dir/$x.mdl "$feats" "$defaultfeats" ark:- $dir/$x.acc2 ) 2>$dir/acc_alimdl.log || exit 1;
  # Update model.
  gmm-est --write-occs=$dir/final.occs --remove-low-count-gaussians=false \
     $dir/$x.mdl $dir/$x.acc2 $dir/$x.alimdl \
     2>$dir/est_alimdl.log  || exit 1;
rm $dir/$x.acc2

( cd $dir; rm final.mdl final.alimdl 2>/dev/null; 
  ln -s $x.mdl final.mdl; ln -s $x.alimdl final.alimdl
  ln -s `basename $cur_lda` final.mat )

