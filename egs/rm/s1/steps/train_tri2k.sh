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

# This (train_tri2k.sh) is training the exponential transform
# after LDA (so the same as LDA+MLLT+ET, since ET includes
# MLLT).

if [ -f path.sh ]; then . path.sh; fi
dir=exp/tri2k
srcdir=exp/tri1
srcgraphs="ark:gunzip -c $srcdir/graphs.fsts.gz|"
srcmodel=$srcdir/final.mdl
dim=40
scale_opts="--transition-scale=1.0 --acoustic-scale=0.1 --self-loop-scale=0.1"
# The spk2utt_opt uses a subset of utterances that we create; this is only
# needed by programs that use the subset.
spk2utt_opt=--spk2utt=ark:$dir/spk2utt
# the utt2spk opt is used by programs that use all the data so give
# it the original utt2spk file.
utt2spk_opt=--utt2spk=ark:data/train.utt2spk
normtype=mean # et option; could be mean, or none

numiters=30    # Number of iterations of training
maxiterinc=20 # Last iter to increase #Gauss on.
numiters_et=15 # Before this, update et.
numleaves=1500
numgauss=$numleaves
totgauss=7000 # Target #Gaussians
incgauss=$[($totgauss-$numgauss)/$maxiterinc] # per-iter increment for #Gauss
realign_iters="10 15 20 25";
silphonelist=`cat data/silphones.csl`

nutt=15 # Use at most 15 utterances from each speaker for
# estimating transforms, and A and B (will use all the data
# for estimating the model though, so be careful: we're
# not always using the lists in $dir).

mkdir -p $dir
cp $srcdir/topo $dir


awk '{ printf("%s ",$1); for(x=2; x<=NF&&x<='$nutt'+1;x++)
    {  printf("%s ", $x); } printf("\n"); }' <data/train.spk2utt >$dir/spk2utt
scripts/spk2utt_to_utt2spk.pl < $dir/spk2utt > $dir/utt2spk
cat $dir/utt2spk | awk '{print $1}' > $dir/uttlist
scripts/filter_scp.pl $dir/uttlist <data/train.scp >$dir/train.scp


srcfeats="ark,s,cs:add-deltas scp:data/train.scp ark:- |"

# For now, there is no subsetting.
basefeats="ark,s,cs:splice-feats scp:data/train.scp ark:- | transform-feats $dir/lda.mat ark:- ark:- |"
## The following two variables will get changed in the script.
feats="$basefeats"



echo "aligning all training data"
gmm-align-compiled  $scale_opts --beam=8 --retry-beam=40  $srcmodel "$srcgraphs" \
       "$srcfeats" ark,t:$dir/0.ali 2> $dir/align.0.log || exit 1;


echo "computing LDA transform"
( ali-to-post ark:$dir/0.ali ark:- | \
  weight-silence-post 0.0 $silphonelist $srcmodel ark:- ark:- | \
  acc-lda $srcmodel "ark:scripts/subset_scp.pl 800 data/train.scp | splice-feats scp:- ark:- |" \
    ark:- $dir/lda.acc ) 2>$dir/lda_acc.log || exit 1

est-lda --dim=$dim $dir/lda.mat $dir/lda.acc 2>$dir/lda_est.log || exit 1

acc-tree-stats  --ci-phones=$silphonelist $srcmodel "$feats" ark:$dir/0.ali \
    $dir/treeacc 2> $dir/acc.tree.log  || exit 1;

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

# Convert alignments generated from previous model, to use as initial alignments.

rm $dir/treeacc

convert-ali  $srcmodel $dir/1.mdl $dir/tree ark:$dir/0.ali ark:$dir/cur.ali 
   2>$dir/convert.log  || exit 1

rm $dir/0.ali

# Make training graphs
echo "Compiling training graphs"
compile-train-graphs $dir/tree $dir/1.mdl  data/L.fst ark:data/train.tra \
   "ark:|gzip -c >$dir/graphs.fsts.gz" 2>$dir/compile_graphs.log || exit 1 

gmm-init-et --normalize-type=$normtype --binary=false --dim=$dim $dir/1.et 2>$dir/init_et.log || exit 1

x=1
while [ $x -lt $numiters ]; do
   x1=$[$x+1]; 
   echo pass $x
   if echo $realign_iters | grep -w $x >/dev/null; then
     echo "Aligning data"
     gmm-align-compiled $scale_opts --beam=8 --retry-beam=40 $dir/$x.mdl \
             "ark:gunzip -c $dir/graphs.fsts.gz|" "$feats" \
             ark:$dir/cur.ali 2> $dir/align.$x.log || exit 1;
   fi

   if [ $x -lt $numiters_et ]; then
     # Work out current transforms:
   ( ali-to-post ark:$dir/cur.ali ark:- | \
     weight-silence-post 0.0 $silphonelist $dir/$x.mdl ark:- ark:- | \
     gmm-post-to-gpost $dir/$x.mdl "$feats" ark:- ark:- | \
     gmm-est-et $spk2utt_opt --verbose=1 $dir/$x.mdl $dir/$x.et "$basefeats" \
        ark,s,cs:- ark:$dir/$x.trans ark,t:$dir/$x.warp ) 2> $dir/trans.$x.log || exit 1;
  
     # Remove previous transforms, if present. 
     if [ $x -gt 1 ]; then rm $dir/$[$x-1].trans; fi

     # Now change $feats to correspond to the transformed features.  We compose the
     # transforms themselves (it's more efficient than transforming the features
     # twice).
     feats="ark:splice-feats scp:data/train.scp ark:- | transform-feats $dir/lda.mat ark:- ark:- | transform-feats $utt2spk_opt ark:$dir/$x.trans ark:- ark:- |"
   fi 

   # Accumulate stats to update model:
   gmm-acc-stats-ali $dir/$x.mdl "$feats" ark:$dir/cur.ali $dir/$x.acc 2>$dir/gmm_acc.$x.log || exit 1;
   # Update model.
   gmm-est --mix-up=$numgauss $dir/$x.mdl $dir/$x.acc $dir/$x1.mdl \
        2>$dir/gmm_est.$x.log || exit 1;

   rm $dir/$x.acc $dir/$x.mdl


   if [ $x -lt $numiters_et ]; then
     # Alternately estimate either A or B.
     if [ $[$x%2] == 0 ]; then  # Estimate A:
     ( ali-to-post ark:$dir/cur.ali ark:- | \
       weight-silence-post 0.0 $silphonelist $dir/$x1.mdl ark:- ark:- | \
       gmm-post-to-gpost $dir/$x1.mdl "$feats" ark:- ark:- | \
       gmm-et-acc-a $spk2utt_opt --verbose=1 $dir/$x1.mdl $dir/$x.et "$basefeats" ark,s,cs:- $dir/$x.et_acc_a ) 2> $dir/acc_a.$x.log || exit 1;
       gmm-et-est-a --verbose=1 $dir/$x.et $dir/$x1.et $dir/$x.et_acc_a 2> $dir/update_a.$x.log || exit 1;
       rm $dir/$x.et_acc_a
     else
     ( ali-to-post ark:$dir/cur.ali ark:- | \
       weight-silence-post 0.0 $silphonelist $dir/$x1.mdl ark:- ark:- | \
       gmm-post-to-gpost $dir/$x1.mdl "$feats" ark:- ark:- | \
       gmm-et-acc-b $spk2utt_opt --verbose=1 $dir/$x1.mdl $dir/$x.et "$basefeats" ark,s,cs:- ark:$dir/$x.trans ark:$dir/$x.warp $dir/$x.et_acc_b ) 2> $dir/acc_b.$x.log || exit 1;
       gmm-et-est-b --verbose=1 $dir/$x.et $dir/$x1.et $dir/$x.mat $dir/$x.et_acc_b 2> $dir/update_b.$x.log || exit 1;
       rm $dir/$x.et_acc_b
       # Careful!: gmm-transform-means here changes $x1.mdl in-place. 
       gmm-transform-means $dir/$x.mat $dir/$x1.mdl $dir/$x1.mdl 2> $dir/transform_means.$x.log
     fi   
   fi
   if [ $x -le $maxiterinc ]; then
      numgauss=$[$numgauss+$incgauss];
   fi
   x=$[$x+1];
done


gmm-et-get-b $dir/$numiters_et.et $dir/B.mat 2>$dir/get_b.log || exit 1
compose-transforms $dir/B.mat $dir/lda.mat $dir/default.mat 2>>$dir/get_b.log || exit 1
defaultfeats="ark,s,cs:splice-feats scp:data/train.scp ark:- | transform-feats $dir/default.mat ark:- ark:- |"

# Accumulate stats for "alignment model" which is as the model but with
# the default features (shares Gaussian-level alignments).
( ali-to-post ark:$dir/cur.ali ark:-  | \
  gmm-acc-stats-twofeats $dir/$x.mdl "$feats" "$defaultfeats" ark:- $dir/$x.acc2 ) 2>$dir/acc_alimdl.log || exit 1;
  # Update model.
gmm-est --write-occs=$dir/final.occs --remove-low-count-gaussians=false $dir/$x.mdl $dir/$x.acc2 $dir/$x.alimdl \
      2>$dir/est_alimdl.log  || exit 1;
rm $dir/$x.acc2


# The following files may be useful for display purposes.
for n in 1 2 3 4 5 6 7 8 9 10 11 12 13 14; do
 cat $dir/$n.warp | scripts/process_warps.pl data/spk2gender.map > $dir/warps.$n
done

( cd $dir; rm final.mdl 2>/dev/null; 
  ln -s $x.mdl final.mdl; ln -s $x.alimdl final.alimdl;
  ln -s $numiters_et.et final.et
  ln -s $[$numiters_et-1].trans final.trans )
 

