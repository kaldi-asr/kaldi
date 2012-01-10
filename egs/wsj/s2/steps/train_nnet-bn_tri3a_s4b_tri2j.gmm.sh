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


# tri2j is triple-deltas + LDA + MLLT/STC
#
# See tri3j for a full-size (si-284) system that builds from this.

if [ -f path.sh ]; then . path.sh; fi

dir=exp/nnet-bn_tri3a_s4b_tri2j.gmm
srcdir=exp/tri3a
srcmodel=$srcdir/final.mdl
scale_opts="--transition-scale=1.0 --acoustic-scale=0.1 --self-loop-scale=0.1"

### GMM training config
numiters=35
maxiterinc=20 # By this iter, we have all the Gaussians.
realign_iters="10 20 30"; 
mllt_iters="2 4 6 12"; 
randprune=5.0 # Use about 1/5 of frames.. (for MLLT, not used here..)
numleaves=4200
numgauss=20000 # initial num-gauss smallish so that transform-training
              # code (when we modify this script) is a bit faster.
totgauss=40000 # Total num-gauss
incgauss=$[($totgauss-$numgauss)/$maxiterinc] # per-iter increment for #Gauss

silphonelist=`cat data/silphones.csl`

mkdir -p $dir
cp $srcdir/train.scp $dir
cp $srcdir/train.tra $dir

scripts/split_scp.pl $dir/train{,1,2,3}.scp
scripts/split_scp.pl $dir/train{,1,2,3}.tra


###### cut the BN-network, get feature transform!
nndir=exp/nnet-bn_tri3a_s4b_net
bndim=30
nnet=$dir/nnet_fwd.nnet
cat $nndir/final.nnet | awk '{ if(match($0,/<biasedlinearity> [0-9]+ '$bndim'/)) {stop=1;} if(stop==0) {print;}}' > $nnet


###### Dump the BN features
bnfeats="ark:splice-feats --print-args=false --left-context=15 --right-context=15 scp:$dir/train.scp ark:- |"
bnfeats="$bnfeats transform-feats $nndir/lda.mat ark:- ark:- |"
bnfeats="$bnfeats apply-cmvn --print-args=false --norm-vars=false ark:$nndir/cmn.ark ark:- ark:- |"
bnfeats="$bnfeats apply-cmvn --print-args=false --norm-vars=true $nndir/global_cvn.mat ark:- ark:- |"

if [ ! -f $dir/bnfeats.scp ]; then
  echo dumping bottleneck features $dir/bnfeats.ark,$dir/bnfeats.scp
  nnet-forward $nnet "$bnfeats" ark,scp:$dir/bnfeats.ark,$dir/bnfeats.scp 2> $dir/nnet-forward.log || touch $dir/.error
else
  echo using pre-dumped bottleneck features $dir/bnfeats.ark,$dir/bnfeats.scp
fi
scripts/split_scp.pl $dir/bnfeats{,1,2,3}.scp
######


srcfeats="ark,s,cs:add-deltas --print-args=false scp:$dir/train.scp ark:- |"
feats="ark,s,cs:add-deltas --delta-order=1 --print-args=false scp:$dir/bnfeats.scp ark:- | transform-feats $dir/lda.mat ark:- ark:- |"
for n in 1 2 3; do
   srcfeatspart[$n]="ark,s,cs:add-deltas --print-args=false scp:$dir/train${n}.scp ark:- |"
   rawfeatspart[$n]="ark,s,cs:add-deltas --delta-order=1 --print-args=false scp:$dir/bnfeats${n}.scp ark:- |"
   featspart[$n]="ark,s,cs:add-deltas --delta-order=1 --print-args=false scp:$dir/bnfeats${n}.scp ark:- | transform-feats $dir/lda.mat ark:- ark:- |"
done

cp $srcdir/topo $dir

# Align all training data using old model (and old graphs, since we
# use the same data-subset as last time). 

echo "Aligning all training data"

rm -f $dir/.error


for n in 1 2 3; do
[ -f $dir/0.${n}.ali.gz ] || gmm-align-compiled $scale_opts --beam=8 --retry-beam=40 $srcmodel \
       "ark:gunzip -c $srcdir/graphs${n}.fsts.gz|" "${srcfeatspart[$n]}" \
       "ark:|gzip -c >$dir/0.${n}.ali.gz" \
           2> $dir/align.0.${n}.log || touch $dir/.error &
done
wait;
[ -f $dir/.error ] &&  echo alignment error RE old system && exit 1

# Now accumulate LDA stats.  Only use the 1st subset of the data,
# since there are not many parameters to estimate.

echo Accumulating LDA stats
for n in 1 2 3; do
 [ -f $dir/lda$n.acc ] || ( ali-to-post "ark:gunzip -c $dir/0.$n.ali.gz|" ark:- | \
    weight-silence-post 0.0 $silphonelist $srcmodel ark:- ark:- | \
    acc-lda $srcmodel "${rawfeatspart[$n]}" \
     ark:- $dir/lda$n.acc ) 2>$dir/lda_acc.$n.log || touch $dir/.error &
done
wait

[ -f $dir/.error ] &&  echo lda-acc error && exit 1

echo Estimating LDA transform
est-lda $dir/lda.mat $dir/lda?.acc  2>$dir/lda_est.log || exit 1


echo Accumulating tree stats
acc-tree-stats  --ci-phones=$silphonelist $srcmodel "$feats" "ark:gunzip -c $dir/0.?.ali.gz|" $dir/treeacc 2> $dir/acc.tree.log  || exit 1;


# The next few commands are involved with making the questions
# for tree clustering.  The extra complexity vs. the RM recipe has
# to do with the desire to ask questions about
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

gmm-init-model  --write-occs=$dir/1.occs  \
    $dir/tree $dir/treeacc $dir/topo $dir/1.mdl 2> $dir/init_model.log || exit 1;

gmm-mixup --mix-up=$numgauss $dir/1.mdl $dir/1.occs $dir/1.mdl \
    2>$dir/mixup.log || exit 1;


rm $dir/treeacc $dir/1.occs


# Convert alignments generated from previous model, to use as initial alignments.
echo Converting alignments
for n in 1 2 3; do
  convert-ali  $srcmodel $dir/1.mdl $dir/tree \
      "ark:gunzip -c $dir/0.$n.ali.gz|" \
      "ark:|gzip -c > $dir/cur$n.ali.gz" 2>$dir/convert.$n.log || exit 1;
done
rm $dir/0.?.ali.gz



# Make training graphs
echo "Compiling training graphs"

rm -f $dir/.error
for n in 1 2 3; do
  compile-train-graphs $dir/tree $dir/1.mdl  data/L.fst ark:$dir/train${n}.tra \
     "ark:|gzip -c > $dir/graphs${n}.fsts.gz" \
     2>$dir/compile_graphs.${n}.log || touch $dir/.error &
done

wait
[ -f $dir/.error ] &&  echo compile-graphs error && exit 1

cur_lda=$dir/lda.mat
x=1
while [ $x -lt $numiters ]; do
   echo "Pass $x"
   if echo $realign_iters | grep -w $x >/dev/null; then
     echo "Aligning data"
     rm -f $dir/.error
     for n in 1 2 3; do
       gmm-align-compiled $scale_opts --beam=8 --retry-beam=40 $dir/$x.mdl \
           "ark:gunzip -c $dir/graphs${n}.fsts.gz|" "${featspart[$n]}" \
           "ark:|gzip -c >$dir/cur${n}.ali.gz" 2> $dir/align.$x.$n.log \
             || touch $dir/.error &
     done
     wait 
     [ -f $dir/.error ] && echo error aligning data && exit 1
   fi
   if echo $mllt_iters | grep -w $x >/dev/null; then # Do MLLT update.
     # Use the 1st of the three subsets... no need to use more as few params to update
     echo "MLLT"
     for n in 1 2 3; do 
      ( ali-to-post "ark:gunzip -c $dir/cur$n.ali.gz|" ark:- | \
        weight-silence-post 0.0 $silphonelist $dir/$x.mdl ark:- ark:- | \
        gmm-acc-mllt --rand-prune=$randprune --binary=false $dir/$x.mdl \
          "${featspart[$n]}" ark:- $dir/$x.$n.macc ) 2> $dir/macc.$x.$n.log \
          || touch $dir/.error &
     done
     wait
     [ -f $dir/.error ] && echo error accumulating MLLT stats && exit 1

     est-mllt $dir/$x.mat.new $dir/$x.?.macc 2> $dir/mupdate.$x.log || exit 1;
     rm $dir/$x.?.macc
     gmm-transform-means --binary=false $dir/$x.mat.new $dir/$x.mdl $dir/$[$x+1].mdl 2> $dir/transform_means.$x.log || exit 1;
     compose-transforms --print-args=false $dir/$x.mat.new $cur_lda $dir/$x.mat || exit 1;

     cur_lda=$dir/$x.mat
     feats="ark,s,cs:add-deltas --delta-order=1 --print-args=false scp:$dir/bnfeats.scp ark:- | transform-feats $cur_lda ark:- ark:- |" 
     for n in 1 2 3; do
       featspart[$n]="ark,s,cs:add-deltas --delta-order=1 --print-args=false scp:$dir/bnfeats$n.scp ark:- | transform-feats $cur_lda ark:- ark:- |" 
     done
   else # do GMM update
     gmm-acc-stats-ali --binary=false $dir/$x.mdl "$feats" \
     "ark,s,cs:gunzip -c $dir/cur?.ali.gz|" $dir/$x.acc 2> $dir/acc.$x.log  || exit 1;
     gmm-est --write-occs=$dir/$[$x+1].occs --mix-up=$numgauss $dir/$x.mdl $dir/$x.acc $dir/$[$x+1].mdl 2> $dir/update.$x.log || exit 1;
   fi
   rm $dir/$x.mdl $dir/$x.acc $dir/$x.occs 2>/dev/null
   
   if [ $x -le $maxiterinc ]; then 
      numgauss=$[$numgauss+$incgauss];
   fi
   x=$[$x+1];
done


( cd $dir; rm final.{mdl,mat} 2>/dev/null; 
  ln -s $x.mdl final.mdl; 
  ln -s `basename $cur_lda` final.mat )
