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

# This (tri2m) is as tri2g except based on LDA+MLLT
# features from tri2f.
# We also start from tri2f for initial alignments.

if [ -f path.sh ]; then . path.sh; fi
dir=exp/tri2m
srcdir=exp/tri2f
srcmodel=$srcdir/final.mdl
srcgraphs="ark:gunzip -c $srcdir/graphs.fsts.gz|"
scale_opts="--transition-scale=1.0 --acoustic-scale=0.1 --self-loop-scale=0.1"
numiters=30    # Number of iterations of training
maxiterinc=20 # Last iter to increase #Gauss on.
numleaves=1800
numgauss=$numleaves
totgauss=9000 # Target #Gaussians
incgauss=$[($totgauss-$numgauss)/$maxiterinc] # per-iter increment for #Gauss
silphonelist=`cat data/silphones.csl`
realign_iters="10 15 20 25";  
lvtln_iters="2 4 6 8 12"; # Recompute LVTLN transforms on these iters.
mat=exp/tri2f/final.mat
if [ ! -f $mat ]; then
  echo No input transformation $mat
  exit 1
fi
per_spk=true
compute_vtlnmdl=true # If true, at the end compute a model with actual feature-space
                     # VTLN features.  You can decode with this as an alternative to
                     # final.mdl which takes the LVTLN features.

numfiles=40 # Number of feature files for computing LVTLN transforms.
numclass=31; # Can't really change this without changing the script below
defaultclass=15; # Corresponds to no warping.
# RE "vtln_warp"


if [ $per_spk == "true" ]; then
  spk2utt_opt=--spk2utt=ark:data/train.spk2utt
  utt2spk_opt=--utt2spk=ark:data/train.utt2spk
else
  spk2utt_opt=
  utt2spk_opt=
fi

mkdir -p $dir
cp $srcdir/topo $dir


srcfeats="ark:splice-feats --print-args=false scp:data/train.scp ark:- | transform-feats $mat ark:- ark:- |"
# Will create lvtln.trans below...
feats="ark:splice-feats --print-args=false scp:data/train.scp ark:- | transform-feats $mat ark:- ark:- | transform-feats $utt2spk_opt ark:$dir/cur.trans ark:- ark:- |"

gmm-init-lvtln --dim=40 --num-classes=$numclass --default-class=$defaultclass \
      $dir/0.lvtln 2>$dir/init_lvtln.log || exit 1

featsub="ark:scripts/subset_scp.pl $numfiles data/train.scp | splice-feats scp:- ark:- | transform-feats $mat ark:- ark:- |"

echo "Initializing lvtln transforms."
c=0
while [ $c -lt $numclass ]; do 
  warp=`perl -e 'print 0.85 + 0.01*$ARGV[0];' $c` 
  featsub_warp="ark:scripts/subset_scp.pl $numfiles data_prep/train_wav.scp | compute-mfcc-feats  --vtln-low=100 --vtln-high=-600 --vtln-warp=$warp --config=conf/mfcc.conf scp:- ark:- | splice-feats ark:- ark:- | transform-feats $mat ark:- ark:- |"
  gmm-train-lvtln-special --normalize-var=true $c $dir/0.lvtln $dir/0.lvtln \
    "$featsub" "$featsub_warp" 2> $dir/train_special.$c.log || exit 1;
  c=$[$c+1]
done



# just a single element. :-separated integer list of context-independent
scripts/make_roots.pl --separate data/phones.txt $silphonelist shared split > $dir/roots.txt 2>$dir/roots.log || exit 1;
# script below tells it not to cluster, but here we avoid accumulating
# CD-stats for silence.

echo "aligning all training data"
gmm-align-compiled  $scale_opts --beam=8 --retry-beam=40  $srcmodel \
 "$srcgraphs" "$srcfeats" ark,t:$dir/0.ali 2> $dir/align.0.log || exit 1;


echo "Computing LVTLN transforms (iter 0)"
( ali-to-post ark:$dir/0.ali  ark:- | \
  weight-silence-post 0.0 $silphonelist $srcmodel ark:- ark:- | \
  gmm-post-to-gpost $srcmodel "$srcfeats" ark:- ark:- | \
  gmm-est-lvtln-trans --verbose=1 $spk2utt_opt $srcmodel $dir/0.lvtln \
    "$srcfeats" ark:- ark:$dir/cur.trans ark,t:$dir/0.warp ) 2>$dir/lvtln.0.log || exit 1

acc-tree-stats  --ci-phones=$silphonelist $srcmodel "$feats" ark:$dir/0.ali $dir/treeacc 2> $dir/acc.tree.log  || exit 1;


cat data/phones.txt | awk '{print $NF}' | grep -v -w 0 > $dir/phones.list
cluster-phones $dir/treeacc $dir/phones.list $dir/questions.txt 2> $dir/questions.log || exit 1;
scripts/int2sym.pl data/phones.txt < $dir/questions.txt > $dir/questions_syms.txt
compile-questions $dir/topo $dir/questions.txt $dir/questions.qst 2>$dir/compile_questions.log || exit 1;

scripts/make_roots.pl --separate data/phones.txt $silphonelist shared split > $dir/roots.txt 2>$dir/roots.log || exit 1;

build-tree --verbose=1 --max-leaves=$numleaves \
    $dir/treeacc $dir/roots.txt \
    $dir/questions.qst $dir/topo $dir/tree  2> $dir/train_tree.log || exit 1;

gmm-init-model  --write-occs=$dir/1.occs  \
    $dir/tree $dir/treeacc $dir/topo $dir/1.mdl 2> $dir/init_model.log || exit 1;

rm $dir/treeacc

# Convert alignments generated from monophone model, to use as initial alignments.

convert-ali  $srcmodel $dir/1.mdl $dir/tree ark:$dir/0.ali ark:$dir/cur.ali 2>$dir/convert.log 
  # Debug step only: convert back and check they're the same.
  convert-ali $dir/1.mdl $srcmodel $srcdir/tree ark:$dir/cur.ali ark,t:- \
   2>/dev/null | cmp - $dir/0.ali || exit 1; 

rm $dir/0.ali


# Make training graphs
echo "Compiling training graphs"
compile-train-graphs $dir/tree $dir/1.mdl  data/L.fst ark:data/train.tra \
   "ark:|gzip -c > $dir/graphs.fsts.gz"  2>$dir/compile_graphs.log || exit 1 

cur_lvtln=$dir/0.lvtln
x=1
while [ $x -lt $numiters ]; do
   echo pass $x
   if echo $lvtln_iters | grep -w $x >/dev/null; then
   ( ali-to-post ark:$dir/cur.ali  ark:- | \
     weight-silence-post 0.0 $silphonelist $dir/$x.mdl ark:- ark:- | \
     gmm-post-to-gpost $dir/$x.mdl "$feats" ark:- ark:- | \
     gmm-est-lvtln-trans --verbose=1 $spk2utt_opt $dir/$x.mdl $dir/0.lvtln \
      "$srcfeats" ark:- ark:$dir/tmp.trans ark,t:$dir/$x.warp ) 2>$dir/lvtln.$x.log || exit 1
     cp $dir/$x.warp $dir/cur.warp
     mv $dir/tmp.trans $dir/cur.trans
   fi
   if echo $realign_iters | grep -w $x >/dev/null; then
     echo "Aligning data"
     gmm-align-compiled $scale_opts --beam=8 --retry-beam=40 $dir/$x.mdl \
             "ark:gunzip -c $dir/graphs.fsts.gz|" "$feats" \
             ark:$dir/cur.ali 2> $dir/align.$x.log || exit 1;
   fi
   gmm-acc-stats-ali --binary=false $dir/$x.mdl "$feats" ark:$dir/cur.ali $dir/$x.acc 2> $dir/acc.$x.log  || exit 1;
   gmm-est --write-occs=$dir/$[$x+1].occs --mix-up=$numgauss $dir/$x.mdl $dir/$x.acc $dir/$[$x+1].mdl 2> $dir/update.$x.log || exit 1;
   rm $dir/$x.mdl $dir/$x.acc
   if [ $x -le $maxiterinc ]; then 
      numgauss=$[$numgauss+$incgauss];
   fi
   x=$[$x+1]
done

# Accumulate stats for "alignment model" which is as the model but with
# the baseline features (shares Gaussian-level alignments).
( ali-to-post ark:$dir/cur.ali ark:-  | \
  gmm-acc-stats-twofeats $dir/$x.mdl "$feats" "$srcfeats" ark:- $dir/$x.acc2 ) 2>$dir/acc_alimdl.log || exit 1;
  # Update model.
gmm-est  --remove-low-count-gaussians=false $dir/$x.mdl $dir/$x.acc2 $dir/$x.alimdl \
      2>$dir/est_alimdl.log  || exit 1;
rm $dir/$x.acc2


# The following files contains information that may be useful for display purposes

for n in 0 $lvtln_iters; do
 cat $dir/$n.warp | scripts/process_warps.pl data/spk2gender.map > $dir/warps.$n
done

if [ $compute_vtlnmdl == "true" ]; then
   cat $dir/cur.warp | awk '{print $1, (0.85+0.01*$2);}' > $dir/cur.factor  
   compute-mfcc-feats $utt2spk_opt --vtln-low=100 --vtln-high=-600 --vtln-map=ark:$dir/cur.factor --config=conf/mfcc.conf scp:data_prep/train_wav.scp ark:$dir/tmp.ark 2>$dir/mfcc.log
   vtlnfeats="ark:splice-feats ark:$dir/tmp.ark ark:- | transform-feats $mat ark:- ark:- |"

   # Compute diagonal fMLLR transform to normalize VTLN feats.
  ( ali-to-post ark:$dir/cur.ali ark:-  | \
    weight-silence-post 0.0 $silphonelist $dir/$x.mdl ark:- ark:- | \
    gmm-est-fmllr --fmllr-update-type=diag $spk2utt_opt $dir/$x.mdl "$vtlnfeats" ark,o:- ark:$dir/vtln.trans ) 2>$dir/vtln_fmllr.log  || exit 1;

   vtlnfeats="ark:splice-feats ark:$dir/tmp.ark ark:- | transform-feats $mat ark:- ark:- | transform-feats $utt2spk_opt ark:$dir/vtln.trans ark:- ark:- |"

  ( ali-to-post ark:$dir/cur.ali ark:-  | \
    gmm-acc-stats-twofeats $dir/$x.mdl "$feats" "$vtlnfeats" ark:- $dir/$x.acc3 ) 2>$dir/acc_vtlnmdl.log || exit 1;
  # Update model.
  gmm-est  $dir/$x.mdl $dir/$x.acc3 $dir/$x.vtlnmdl \
      2>$dir/est_vtlnmdl.log  || exit 1;
  rm $dir/$x.acc3
  ln -s $x.vtlnmdl $dir/final.vtlnmdl
  rm $dir/tmp.ark
fi


( cd $dir; rm final.mdl 2>/dev/null; ln -s $x.mdl final.mdl;
  ln -s $x.alimdl final.alimdl;
  ln -s 0.lvtln final.lvtln;
  ln -s cur.trans final.trans )
