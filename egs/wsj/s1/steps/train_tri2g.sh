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


# tri2g is as tri2a but using "linear VTLN" (a VTLN
# substitute); at the end, it converts in one pass to actual VTLN
# features (we use this as an alternative to LVTLN).
# See tri3g for a full-size (si-284) system that builds from this.

if [ -f path.sh ]; then . path.sh; fi

dir=exp/tri2g
srcdir=exp/tri1
srcmodel=$srcdir/final.mdl
scale_opts="--transition-scale=1.0 --acoustic-scale=0.1 --self-loop-scale=0.1"

# This block of parameters relates to LVTLN.
compute_vtlnmdl=true # If true, at the end compute a model with actual feature-space
                     # VTLN features.  You can decode with this as an alternative to
                     # final.mdl which takes the LVTLN features.
dim=39 # the dim of our features.
lvtln_iters="2 4 6 8 12"; # Recompute LVTLN transforms on these iters.
numfiles=40 # Number of feature files for computing LVTLN transforms.
numclass=31; # Can't really change this without changing the script below
defaultclass=15; # Corresponds to no warping.


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

scripts/filter_scp.pl $dir/train.scp data/train_wav.scp > $dir/train_wav.scp
scripts/filter_scp.pl $dir/train.scp data/train.utt2spk > $dir/train.utt2spk

scripts/split_scp.pl --utt2spk=$dir/train.utt2spk $dir/train{,1,2,3}.scp
scripts/split_scp.pl --utt2spk=$dir/train.utt2spk $dir/train_wav{,1,2,3}.scp
scripts/split_scp.pl --utt2spk=$dir/train.utt2spk $dir/train{,1,2,3}.tra
scripts/split_scp.pl --utt2spk=$dir/train.utt2spk $dir/train{,1,2,3}.utt2spk


for n in 1 2 3 ""; do # The "" handles the un-split one.  Creating spk2utt files..
  scripts/utt2spk_to_spk2utt.pl $dir/train$n.utt2spk > $dir/train$n.spk2utt
done

# also see featspart below, used for sub-parts of the features;
# try to keep them in sync.
feats="ark,s,cs:add-deltas --print-args=false scp:$dir/train.scp ark:- | transform-feats --utt2spk=ark:$dir/train.utt2spk \"ark:cat $dir/cur?.trans|\" ark:- ark:- |"
srcfeats="ark,s,cs:add-deltas --print-args=false scp:$dir/train.scp ark:- |"
for n in 1 2 3; do
   featspart[$n]="ark,s,cs:add-deltas --print-args=false scp:$dir/train${n}.scp ark:- | transform-feats --utt2spk=ark:$dir/train.utt2spk ark:$dir/cur$n.trans ark:- ark:- |"
   srcfeatspart[$n]="ark,s,cs:add-deltas --print-args=false scp:$dir/train${n}.scp ark:- |"
done

cp $srcdir/topo $dir



gmm-init-lvtln --dim=$dim --num-classes=$numclass --default-class=$defaultclass \
      $dir/0.lvtln 2>$dir/init_lvtln.log || exit 1

# Small subset of features for initializing the LVTLN.

featsub="ark:scripts/subset_scp.pl $numfiles $dir/train.scp | add-deltas scp:- ark:- |"

echo "Initializing lvtln transforms."
c=0
while [ $c -lt $numclass ]; do 
  warp=`perl -e 'print 0.85 + 0.01*$ARGV[0];' $c` 
  featsub_warp="ark:scripts/subset_scp.pl $numfiles $dir/train_wav.scp | compute-mfcc-feats  --vtln-low=100 --vtln-high=-600 --vtln-warp=$warp --config=conf/mfcc.conf scp:- ark:- | add-deltas ark:- ark:- |"
  gmm-train-lvtln-special --normalize-var=true $c $dir/0.lvtln $dir/0.lvtln \
    "$featsub" "$featsub_warp" 2> $dir/train_special.$c.log || exit 1;
  c=$[$c+1]
done


# Align all training data using old model (and old graphs, since we
# use the same data-subset as last time). 
# Note: a few fail to get aligned here due to the difference between
# per-speaker and per-utterance splitting, but this doesn't really matter.

echo "Aligning all training data"

rm -f $dir/.error
for n in 1 2 3; do
   gmm-align-compiled $scale_opts --beam=8 --retry-beam=40 $srcmodel \
       "ark:gunzip -c $srcdir/graphs${n}.fsts.gz|" "${srcfeatspart[$n]}" \
       "ark:|gzip -c >$dir/0.${n}.ali.gz" \
           2> $dir/align.0.${n}.log || touch $dir/.error &
done
wait;
[ -f $dir/.error ] &&  echo alignment error RE old system && exit 1

echo "Computing LVTLN transforms (iter 0)"
rm -f $dir/.error
for n in 1 2 3; do
 ( ali-to-post "ark:gunzip -c $dir/0.$n.ali.gz|"  ark:- | \
   weight-silence-post 0.0 $silphonelist $srcmodel ark:- ark:- | \
   gmm-post-to-gpost $srcmodel "${srcfeatspart[$n]}" ark:- ark:- | \
   gmm-est-lvtln-trans --verbose=1 --spk2utt=ark:$dir/train$n.spk2utt $srcmodel $dir/0.lvtln \
    "${srcfeatspart[$n]}" ark:- ark:$dir/cur$n.trans ark,t:$dir/0.$n.warp ) \
        2>$dir/lvtln.0.$n.log || touch $dir/.error &
done
wait;
[ -f $dir/.error ] &&  echo error computing LVTLN transforms on iter 0 && exit 1


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
  convert-ali $srcmodel $dir/1.mdl $dir/tree \
      "ark:gunzip -c $dir/0.$n.ali.gz|" \
      "ark:|gzip -c > $dir/cur$n.ali.gz" 2>$dir/convert.$n.log || exit 1;
done
rm $dir/0.?.ali.gz

# Make training graphs
echo "Compiling training graphs"

rm -f $dir/.error
for n in 1 2 3; do
  compile-train-graphs $dir/tree $dir/1.mdl data/L.fst ark:$dir/train${n}.tra \
     "ark:|gzip -c > $dir/graphs${n}.fsts.gz" \
     2>$dir/compile_graphs.${n}.log || touch $dir/.error &
done
wait
[ -f $dir/.error ] &&  echo compile-graphs error && exit 1


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
   if echo $lvtln_iters | grep -w $x >/dev/null; then
     # Work out current transforms (in parallel).
     echo "Computing LVTLN transforms"
     rm -f $dir/.error
     for n in 1 2 3; do
     ( ali-to-post "ark:gunzip -c $dir/cur${n}.ali.gz|" ark:- | \
       weight-silence-post 0.0 $silphonelist $dir/$x.mdl ark:- ark:- | \
       gmm-post-to-gpost $dir/$x.mdl "${featspart[$n]}" ark,o:- ark:- | \
       gmm-est-lvtln-trans --spk2utt=ark:$dir/train$n.spk2utt --verbose=1 $dir/$x.mdl $dir/0.lvtln \
       "${srcfeatspart[$n]}" ark,s,cs:- ark:$dir/tmp$n.trans ark,t:$dir/$x.$n.warp ) \
          2> $dir/trans.$x.$n.log && mv $dir/tmp$n.trans $dir/cur$n.trans \
             || touch $dir/.error &
     done
     wait 
     [ -f $dir/.error ] && echo error aligning data && exit 1
   fi
   gmm-acc-stats-ali --binary=false $dir/$x.mdl "$feats" \
     "ark,s,cs:gunzip -c $dir/cur?.ali.gz|" $dir/$x.acc 2> $dir/acc.$x.log  || exit 1;
   gmm-est --write-occs=$dir/$[$x+1].occs --mix-up=$numgauss $dir/$x.mdl $dir/$x.acc $dir/$[$x+1].mdl 2> $dir/update.$x.log || exit 1;
   rm $dir/$x.mdl $dir/$x.acc $dir/$x.occs 2>/dev/null

   if [ $x -le $maxiterinc ]; then 
      numgauss=$[$numgauss+$incgauss];
   fi
   x=$[$x+1];
done

# Accumulate stats for "alignment model" which is as the model but with
# the baseline features (shares Gaussian-level alignments).
( ali-to-post "ark:gunzip -c $dir/cur?.ali.gz|" ark:- | \
  gmm-acc-stats-twofeats $dir/$x.mdl "$feats" "$srcfeats" ark:- $dir/$x.acc2 ) 2>$dir/acc_alimdl.log || exit 1;
  # Update model.
  gmm-est  --remove-low-count-gaussians=false $dir/$x.mdl $dir/$x.acc2 $dir/$x.alimdl \
      2>$dir/est_alimdl.log  || exit 1;
rm $dir/$x.acc2


if [ $compute_vtlnmdl == "true" ]; then
   iter=`echo 0 $lvtln_iters | awk '{print $NF}'` # last iter we re-estimated LVTLN
   rm -f $dir/.error
   for n in 1 2 3; do
     cat $dir/$iter.$n.warp | awk '{print $1, (0.85+0.01*$2);}' > $dir/cur$n.factor  
     compute-mfcc-feats --utt2spk=ark:$dir/train$n.utt2spk --vtln-low=100 --vtln-high=-600 \
          --vtln-map=ark:$dir/cur$n.factor --config=conf/mfcc.conf \
           scp:$dir/train_wav$n.scp ark:$dir/tmp$n.ark 2>$dir/mfcc.$n.log \
         || touch $dir/.error &
   done
   wait
   [ -f $dir/.error ] && echo error computing VTLN-warped MFCC features && exit 1

   # Compute diagonal fMLLR transform to normalize VTLN feats.
   # (note, this is a bit stronger than the mean-only transform we used for the LVTLN stuff, 
   #  LVTLN also globally normalized the variance of each warp factor, so this seems
   #  appropriate). 
   for n in 1 2 3; do
     vtlnfeats="ark:add-deltas ark:$dir/tmp$n.ark ark:- |"
     ( ali-to-post "ark:gunzip -c $dir/cur$n.ali.gz|" ark:-  | \
      weight-silence-post 0.0 $silphonelist $dir/$x.mdl ark:- ark:- | \
      gmm-est-fmllr --fmllr-update-type=diag --spk2utt=ark:$dir/train$n.spk2utt \
         $dir/$x.mdl "$vtlnfeats" ark,o:- ark:$dir/vtln$n.trans ) \
        2>$dir/vtln_fmllr.$n.log  || touch $dir/.error &
   done
   wait
   [ -f $dir/.error ] && echo error computing fMLLR transforms after VTLN && exit 1

   # all the features, with diagonal fMLLR
   vtlnfeats="ark:cat $dir/tmp?.ark | add-deltas ark:- ark:- | transform-feats --utt2spk=ark:$dir/train.utt2spk \"ark:cat $dir/vtln?.trans|\" ark:- ark:- |"

  ( ali-to-post "ark:gunzip -c $dir/cur?.ali.gz|" ark:-  | \
    gmm-acc-stats-twofeats $dir/$x.mdl "$feats" "$vtlnfeats" ark,s,cs:- $dir/$x.acc3 ) 2>$dir/acc_vtlnmdl.log || exit 1;
  # Update model.
  gmm-est  $dir/$x.mdl $dir/$x.acc3 $dir/$x.vtlnmdl \
      2>$dir/est_vtlnmdl.log  || exit 1;
  rm $dir/$x.acc3
  rm $dir/final.alimdl 2>/dev/null
  ln -s $x.vtlnmdl $dir/final.vtlnmdl
  rm $dir/tmp?.ark
fi



# The following files may be be useful for display purposes.
for y in lvtln_iters; do
  cat $dir/$y.?.warp | scripts/process_warps.pl data/spk2gender.map > $dir/warps.$y
  y=$[$y+1]
done


( cd $dir; rm final.{mdl,alimdl,et} 2>/dev/null; 
  ln -s $x.mdl final.mdl; ln -s $x.alimdl final.alimdl;
  ln -s $numiters_et.et final.et )
