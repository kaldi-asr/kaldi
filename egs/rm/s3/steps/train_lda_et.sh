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
# Triphone model training, using cepstral mean normalization plus
# splice-9-frames.  It starts from an existing directory (e.g.
# exp/tri1), supplied as an argument, which is assumed to be built using
# cepstral mean subtraction plus delta features.

if [ $# != 4 ]; then
   echo "Usage: steps/train_lda_et.sh <data-dir> <lang-dir> <ali-dir> <exp-dir>"
   echo " e.g.: steps/train_lda_et.sh data/train data/lang exp/tri1_ali exp/tri2c"
   exit 1;
fi

if [ -f path.sh ]; then . path.sh; fi

data=$1
lang=$2
alidir=$3
dir=$4


numiters_et=15 # Before this, update et parameters.
normtype=offset # et option; could be offset [recommended], or none

scale_opts="--transition-scale=1.0 --acoustic-scale=0.1 --self-loop-scale=0.1"
realign_iters="5 10 15 20";  
silphonelist=`cat $lang/silphones.csl`
numiters=25    # Number of iterations of training
maxiterinc=15 # Last iter to increase #Gauss on.
numleaves=1800 # target num-leaves in tree building.
numgauss=$[$numleaves + $numleaves/2];  # starting num-Gauss.
     # Initially mix up to avg. 1.5 Gauss/state ( a bit more
     # than this, due to state clustering... then slowly mix 
     # up to final amount.
totgauss=9000 # Target #Gaussians
incgauss=$[($totgauss-$numgauss)/$maxiterinc] # per-iter increment for #Gauss
randprune=4.0

mkdir -p $dir $dir/warps

# This variable gets overwritten in this script.
basefeats="ark,s,cs:apply-cmvn --norm-vars=false --utt2spk=ark:$data/utt2spk ark:$alidir/cmvn.ark scp:$data/feats.scp ark:- | splice-feats ark:- ark:- | transform-feats $dir/lda.mat ark:- ark:- |"
feats="$basefeats"
splicedfeats="ark,s,cs:apply-cmvn --norm-vars=false --utt2spk=ark:$data/utt2spk ark:$alidir/cmvn.ark scp:$data/feats.scp ark:- | splice-feats ark:- ark:- |"

# compute integer form of transcripts.
scripts/sym2int.pl --ignore-first-field $lang/words.txt < $data/text > $dir/train.tra \
  || exit 1;

echo "Accumulating LDA statistics."

( ali-to-post ark:$alidir/ali ark:- | \
   weight-silence-post 0.0 $silphonelist $alidir/final.mdl ark:- ark:- | \
   acc-lda --rand-prune=$randprune $alidir/final.mdl "$splicedfeats" ark,s,cs:- $dir/lda.acc ) \
   2>$dir/lda_acc.log
est-lda $dir/lda.mat $dir/lda.acc 2>$dir/lda_est.log

cur_lda=$dir/lda.mat

echo "Accumulating tree stats"

acc-tree-stats  --ci-phones=$silphonelist $alidir/final.mdl "$feats" \
    ark:$alidir/ali $dir/treeacc 2> $dir/acc.tree.log  || exit 1;


echo "Computing questions for tree clustering"

cat $lang/phones.txt | awk '{print $NF}' | grep -v -w 0 > $dir/phones.list
cluster-phones $dir/treeacc $dir/phones.list $dir/questions.txt 2> $dir/questions.log || exit 1;
scripts/int2sym.pl $lang/phones.txt < $dir/questions.txt > $dir/questions_syms.txt
compile-questions $lang/topo $dir/questions.txt $dir/questions.qst 2>$dir/compile_questions.log || exit 1;

# Have to make silence root not-shared because we will not split it.
scripts/make_roots.pl --separate $lang/phones.txt $silphonelist shared split \
    > $dir/roots.txt 2>$dir/roots.log || exit 1;


echo "Building tree"
build-tree --verbose=1 --max-leaves=$numleaves \
    $dir/treeacc $dir/roots.txt \
    $dir/questions.qst $lang/topo $dir/tree  2> $dir/train_tree.log || exit 1;

gmm-init-model  --write-occs=$dir/1.occs  \
    $dir/tree $dir/treeacc $lang/topo $dir/1.mdl 2> $dir/init_model.log || exit 1;

gmm-mixup --mix-up=$numgauss $dir/1.mdl $dir/1.occs $dir/1.mdl \
   2>$dir/mixup.log || exit 1;

rm $dir/treeacc

# Convert alignments generated from monophone model, to be used as initial alignments.

convert-ali $alidir/final.mdl $dir/1.mdl $dir/tree ark:$alidir/ali ark:$dir/cur.ali 2>$dir/convert.log 
  # Debug step only: convert back and check they're the same.
  convert-ali $dir/1.mdl $alidir/final.mdl $alidir/tree ark:$dir/cur.ali ark:- \
   2>/dev/null | cmp - $alidir/ali || exit 1; 

# Make training graphs
echo "Compiling training graphs"
compile-train-graphs $dir/tree $dir/1.mdl  $lang/L.fst ark:$dir/train.tra \
    "ark:|gzip -c >$dir/graphs.fsts.gz"  2>$dir/compile_graphs.log  || exit 1;

gmm-init-et --normalize-type=$normtype --binary=false --dim=40 $dir/1.et 2>$dir/init_et.log || exit 1

x=1
while [ $x -lt $numiters ]; do
   echo Pass $x
   if echo $realign_iters | grep -w $x >/dev/null; then
     echo "Aligning data"
     gmm-align-compiled $scale_opts --beam=8 --retry-beam=40 $dir/$x.mdl \
             "ark:gunzip -c $dir/graphs.fsts.gz|" "$feats" \
             ark:$dir/cur.ali 2> $dir/align.$x.log || exit 1;
   fi
   if [ $x -lt $numiters_et ]; then
     echo "Re-estimating ET transforms"
   ( ali-to-post ark:$dir/cur.ali ark:- | \
     weight-silence-post 0.0 $silphonelist $dir/$x.mdl ark:- ark:- | \
     rand-prune-post $randprune ark:- ark:- | \
     gmm-post-to-gpost $dir/$x.mdl "$feats" ark:- ark:- | \
     gmm-est-et --spk2utt=ark:$data/spk2utt $dir/$x.mdl $dir/$x.et "$basefeats" \
        ark,s,cs:- ark:$dir/$x.trans ark,t:$dir/warps/$x.warp ) \
     2> $dir/trans.$x.log || exit 1;

     # Remove previous transforms, if present. 
     if [ $x -gt 1 ]; then rm $dir/$[$x-1].trans; fi
     # Set features to include transform.
     feats="$basefeats transform-feats --utt2spk=ark:$data/utt2spk ark:$dir/$x.trans ark:- ark:- |"
   fi

   gmm-acc-stats-ali --binary=false $dir/$x.mdl "$feats" ark:$dir/cur.ali $dir/$x.acc 2> $dir/acc.$x.log || exit 1;
   gmm-est --write-occs=$dir/$[$x+1].occs --mix-up=$numgauss $dir/$x.mdl $dir/$x.acc $dir/$[$x+1].mdl 2> $dir/update.$x.log || exit 1;
   rm $dir/$x.mdl $dir/$x.acc
   rm $dir/$x.occs 

   x1=$[$x+1]
   if [ $x -lt $numiters_et ]; then
     # Alternately estimate either A or B.
     if [ $[$x%2] == 0 ]; then  # Estimate A:
       ( ali-to-post ark:$dir/cur.ali ark:- | \
         weight-silence-post 0.0 $silphonelist $dir/$x1.mdl ark:- ark:- | \
         rand-prune-post $randprune ark:- ark:- | \
         gmm-post-to-gpost $dir/$x1.mdl "$feats" ark:- ark:- | \
         gmm-et-acc-a --spk2utt=ark:$data/spk2utt --verbose=1 $dir/$x1.mdl $dir/$x.et "$basefeats" \
              ark,s,cs:- $dir/$x.et_acc_a ) 2> $dir/acc_a.$x.log || exit 1;
       gmm-et-est-a --verbose=1 $dir/$x.et $dir/$x1.et $dir/$x.et_acc_a 2> $dir/update_a.$x.log || exit 1;
       rm $dir/$x.et_acc_a
     else
      ( ali-to-post ark:$dir/cur.ali ark:- | \
        weight-silence-post 0.0 $silphonelist $dir/$x1.mdl ark:- ark:- | \
        gmm-acc-mllt --rand-prune=$randprune $dir/$x1.mdl "$feats" ark:- \
           $dir/$x.mllt_acc ) 2> $dir/acc_b.$x.log || exit 1;
        est-mllt $dir/$x.mat $dir/$x.mllt_acc 2> $dir/update_b.$x.log || exit 1;
       gmm-et-apply-c $dir/$x.et $dir/$x.mat $dir/$x1.et 2>>$dir/update_b.$x.log || exit 1;
       gmm-transform-means $dir/$x.mat $dir/$x1.mdl $dir/$x1.mdl 2>> $dir/update_b.$x.log || exit 1;
       # Modify current transforms by premultiplying by C.
       compose-transforms $dir/$x.mat ark:$dir/$x.trans ark:$dir/tmp.trans 2>> $dir/update_b.$x.log || exit 1;
       mv $dir/tmp.trans $dir/$x.trans
       rm $dir/$x.mat
     fi   
   fi

   if [[ $x -le $maxiterinc ]]; then 
      numgauss=$[$numgauss+$incgauss];
   fi
   x=$[$x+1];
done



# Write out the B matrix which we will combine with LDA to get
# final.mat; and write out final.et which is the current final et
# but with B set to unity (since it's now part of final.mat).
# This is just more convenient going forward, since the "default features"
# (i.e. when speaker factor equals zero) are now the same as the
# features that the ET acts on.

gmm-et-get-b $dir/$numiters_et.et $dir/B.mat $dir/final.et 2>$dir/get_b.log || exit 1

compose-transforms $dir/B.mat $dir/lda.mat $dir/final.mat 2>>$dir/get_b.log || exit 1

defaultfeats="$basefeats transform-feats $dir/B.mat ark:- ark:- |"

# Accumulate stats for "alignment model" which is as the model but with
# the default features (shares Gaussian-level alignments).
( ali-to-post ark:$dir/cur.ali ark:-  | \
  gmm-acc-stats-twofeats $dir/$x.mdl "$feats" "$defaultfeats" ark:- $dir/$x.acc2 ) 2>$dir/acc_alimdl.log || exit 1;
  # Update model.
 gmm-est --write-occs=$dir/final.occs --remove-low-count-gaussians=false $dir/$x.mdl $dir/$x.acc2 $dir/$x.alimdl \
      2>$dir/est_alimdl.log  || exit 1;
rm $dir/$x.acc2

# The following files may be useful for display purposes.
for y in 1 2 3 4 5 6 7 8 9 10 11 12 13 14; do
  cat $dir/warps/$y.warp | scripts/process_warps.pl $data/spk2gender > $dir/warps/$y.warp_info
done

( cd $dir; rm final.mdl 2>/dev/null; 
  ln -s $x.mdl final.mdl; ln -s $x.alimdl final.alimdl;
  ln -s $[$numiters_et-1].trans final.trans )

echo Done
