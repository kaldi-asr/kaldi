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


# tri2b is as tri2a but using the "exponential transform" (a kind of VTLN
# substitute).
# See tri3b for a full-size (si-284) system that builds from this.

if [ -f path.sh ]; then . path.sh; fi

dir=exp/tri2b
srcdir=exp/tri1
srcmodel=$srcdir/final.mdl
scale_opts="--transition-scale=1.0 --acoustic-scale=0.1 --self-loop-scale=0.1"

# The 3 following settings relate to ET.
dim=39 # the dim of our features.
normtype=mean
numiters_et=15 # Before this, update et.

numiters=35
maxiterinc=20 # By this iter, we have all the Gaussians.
realign_iters="10 20 30"; 
numleaves=2000
numgauss=2000 # initial num-gauss smallish so that transform-training
              # code (when we modify this script) is a bit faster.
totgauss=10000 # Total num-gauss
incgauss=$[($totgauss-$numgauss)/$maxiterinc] # per-iter increment for #Gauss

silphonelist=`cat data/silphones.csl`

mkdir -p $dir
cp $srcdir/train.scp $dir
cp $srcdir/train.tra $dir

scripts/filter_scp.pl $dir/train.scp data/train.utt2spk > $dir/train.utt2spk

scripts/split_scp.pl --utt2spk=$dir/train.utt2spk $dir/train{,1,2,3}.scp
scripts/split_scp.pl --utt2spk=$dir/train.utt2spk $dir/train{,1,2,3}.tra
scripts/split_scp.pl --utt2spk=$dir/train.utt2spk $dir/train{,1,2,3}.utt2spk

for n in 1 2 3 ""; do # The "" handles the un-split one.  Creating spk2utt files..
  scripts/utt2spk_to_spk2utt.pl $dir/train$n.utt2spk > $dir/train$n.spk2utt
done

# also see featspart below, used for sub-parts of the features;
# try to keep them in sync.
feats="ark,s,cs:add-deltas --print-args=false scp:$dir/train.scp ark:- |"
origfeats="ark,s,cs:add-deltas --print-args=false scp:$dir/train.scp ark:- |"
for n in 1 2 3; do
   featspart[$n]="ark,s,cs:add-deltas --print-args=false scp:$dir/train${n}.scp ark:- |"
   origfeatspart[$n]="ark,s,cs:add-deltas --print-args=false scp:$dir/train${n}.scp ark:- |"
done

cp $srcdir/topo $dir

# Align all training data using old model (and old graphs, since we
# use the same data-subset as last time). 
# Note: a few fail to get aligned here due to the difference between
# per-speaker and per-utterance splitting, but this doesn't really matter.

echo "Aligning all training data"

rm -f $dir/.error
for n in 1 2 3; do
   gmm-align-compiled $scale_opts --beam=8 --retry-beam=40 $srcmodel \
       "ark:gunzip -c $srcdir/graphs${n}.fsts.gz|" "${featspart[$n]}" \
       "ark:|gzip -c >$dir/0.${n}.ali.gz" \
           2> $dir/align.0.${n}.log || touch $dir/.error &
done
wait;
[ -f $dir/.error ] &&  echo alignment error RE old system && exit 1


acc-tree-stats  --ci-phones=$silphonelist $srcmodel "$feats" "ark:gunzip -c $dir/0.?.ali.gz|" $dir/treeacc 2> $dir/acc.tree.log  || exit 1;


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

gmm-init-model  --write-occs=$dir/1.occs  \
    $dir/tree $dir/treeacc $dir/topo $dir/1.mdl 2> $dir/init_model.log || exit 1;

gmm-mixup --mix-up=$numgauss $dir/1.mdl $dir/1.occs $dir/1.mdl \
    2>$dir/mixup.log || exit 1;


rm $dir/treeacc $dir/1.occs


# Convert alignments generated from previous model, to use as initial alignments.

for n in 1 2 3; do
  convert-ali  $srcmodel $dir/1.mdl $dir/tree \
      "ark:gunzip -c $dir/0.$n.ali.gz|" \
      "ark:|gzip -c > $dir/cur$n.ali.gz" 2>$dir/convert.$n.log  || exit 1;
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

gmm-init-et --normalize-type=$normtype --binary=false --dim=$dim $dir/1.et 2>$dir/init_et.log || exit 1

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
   if [ $x -lt $numiters_et ]; then
     # Work out current transforms (in parallel).
     echo "Computing ET transforms"
     rm -f $dir/.error
     for n in 1 2 3; do
     ( ali-to-post "ark:gunzip -c $dir/cur${n}.ali.gz|" ark:- | \
       weight-silence-post 0.0 $silphonelist $dir/$x.mdl ark:- ark:- | \
       gmm-post-to-gpost $dir/$x.mdl "${featspart[$n]}" ark,o:- ark:- | \
       gmm-est-et --spk2utt=ark:$dir/train$n.spk2utt --verbose=1 $dir/$x.mdl $dir/$x.et \
       "${origfeatspart[$n]}" ark,s,cs:- ark:$dir/$x.$n.trans ark,t:$dir/$x.$n.warp ) \
          2> $dir/trans.$x.$n.log || touch $dir/.error &
     done
     wait 
     [ -f $dir/.error ] && echo error aligning data && exit 1

     # Remove previous transforms, if present. 
     if [ $x -gt 1 ]; then rm $dir/$[$x-1].?.trans; fi
     # Now change $feats and $featspart to correspond to the transformed features. 
     feats="ark,s,cs:add-deltas scp:$dir/train.scp ark:- | transform-feats --utt2spk=ark:$dir/train.utt2spk \"ark,s,cs:cat $dir/$x.?.trans|\" ark:- ark:- |"
     for n in 1 2 3; do
       featspart[$n]="ark,s,cs:add-deltas scp:$dir/train$n.scp ark:- | transform-feats --utt2spk=ark:$dir/train$n.utt2spk ark:$dir/$x.$n.trans ark:- ark:- |"
     done
   fi
   gmm-acc-stats-ali --binary=false $dir/$x.mdl "$feats" \
     "ark,s,cs:gunzip -c $dir/cur?.ali.gz|" $dir/$x.acc 2> $dir/acc.$x.log  || exit 1;
   gmm-est --write-occs=$dir/$[$x+1].occs --mix-up=$numgauss $dir/$x.mdl $dir/$x.acc $dir/$[$x+1].mdl 2> $dir/update.$x.log || exit 1;
   rm $dir/$x.mdl $dir/$x.acc $dir/$x.occs 2>/dev/null

   if [ $x -lt $numiters_et ]; then
     # Alternately estimate either A or B.

     x1=$[$x+1]
     if [ $[$x%2] == 0 ]; then  # Estimate A:
       for n in 1 2 3; do
       ( ali-to-post "ark:gunzip -c $dir/cur${n}.ali.gz|" ark:- | \
         weight-silence-post 0.0 $silphonelist $dir/$x1.mdl ark:- ark:- | \
         gmm-post-to-gpost $dir/$x1.mdl "${featspart[$n]}" ark:- ark:- | \
         gmm-et-acc-a --spk2utt=ark:$dir/train$n.spk2utt --verbose=1 $dir/$x1.mdl \
             $dir/$x.et "${origfeatspart[$n]}" ark,s,cs:- $dir/$x.$n.et_acc_a ) \
           2> $dir/acc_a.$x.$n.log || touch $dir/.error &
       done
       wait 
       [ -f $dir/.error ] && echo error computing stats to accumulate A && exit 1
       gmm-et-est-a --verbose=1 $dir/$x.et $dir/$x1.et $dir/$x.?.et_acc_a 2> $dir/update_a.$x.log || exit 1;
       rm $dir/$x.?.et_acc_a
     else
       for n in 1 2 3; do
       ( ali-to-post "ark:gunzip -c $dir/cur${n}.ali.gz|" ark:- | \
         weight-silence-post 0.0 $silphonelist $dir/$x1.mdl ark:- ark:- | \
         gmm-acc-mllt $dir/$x1.mdl "${featspart[$n]}" ark,s,cs:- $dir/$x.$n.mllt_acc ) \
          2> $dir/acc_b.$x.$n.log || touch $dir/.error & 
       done
       wait 
       [ -f $dir/.error ] && echo error computing stats for estimating B && exit 1
       est-mllt $dir/$x.mat $dir/$x.{1,2,3}.mllt_acc 2>$dir/update_b.$x.log || exit 1;
       rm $dir/$x.{1,2,3}.mllt_acc
       gmm-et-apply-c $dir/$x.et $dir/$x.mat $dir/$x1.et 2>>$dir/update_b.$x.log || exit 1;
       gmm-transform-means $dir/$x.mat $dir/$x1.mdl $dir/$x1.mdl 2>> $dir/update_b.$x.log || exit 1;
       # Modify current transforms by premultiplying by C.
       for n in 1 2 3; do
         compose-transforms $dir/$x.mat ark:$dir/$x.$n.trans ark:$dir/tmp.trans 2>> $dir/update_b.$x.log || exit 1;
         mv $dir/tmp.trans $dir/$x.$n.trans
       done
       rm $dir/$x.mat
     fi   
   fi


   if [ $x -le $maxiterinc ]; then 
      numgauss=$[$numgauss+$incgauss];
   fi
   x=$[$x+1];
done

# Accumulate stats for "alignment model" which is as the model but with
# the baseline features (shares Gaussian-level alignments).

gmm-et-get-b $dir/$numiters_et.et $dir/default.mat 2>$dir/get_b.log || exit 1

defaultfeats="ark,s,cs:add-deltas scp:$dir/train.scp ark:- | transform-feats $dir/default.mat ark:- ark:- |"


( ali-to-post "ark:gunzip -c $dir/cur?.ali.gz|" ark:- | \
  gmm-acc-stats-twofeats $dir/$x.mdl "$feats" "$defaultfeats" ark:- $dir/$x.acc2 ) 2>$dir/acc_alimdl.log || exit 1;
  # Update model.
  gmm-est  --remove-low-count-gaussians=false $dir/$x.mdl $dir/$x.acc2 $dir/$x.alimdl \
      2>$dir/est_alimdl.log  || exit 1;
rm $dir/$x.acc2

# The following files may be be useful for display purposes.
y=1
while [ $y -lt $numiters_et ]; do
  cat $dir/$y.?.warp | scripts/process_warps.pl data/spk2gender.map > $dir/warps.$y
  y=$[$y+1]
done


( cd $dir; rm final.{mdl,alimdl,et} 2>/dev/null; 
  ln -s $x.mdl final.mdl; ln -s $x.alimdl final.alimdl;
  ln -s $numiters_et.et final.et )
