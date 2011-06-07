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


# tri3k is triphone training with splice-9-frames + LDA + ET (exponential
# transform), using the entire si-284 training set, starting from the
# model in tri2k.  Initializing the states from the previous model's states 
# for faster training.  We use the previous model to get the speaker transforms,
# and do this only once (we use the alignment model).
# This script uses (about) 3 CPUs.  

if [ -f path.sh ]; then . path.sh; fi

dir=exp/tri3k
srcdir=exp/tri2k
srcmodel=$srcdir/final.mdl
et=$srcdir/final.et
ldamat=$srcdir/lda.mat
defaultmat=$srcdir/default.mat # with the "default" exponential transform, used by alignment model.
srcalimodel=$srcdir/final.alimdl
scale_opts="--transition-scale=1.0 --acoustic-scale=0.1 --self-loop-scale=0.1"

numiters=20
maxiterinc=10 # By this iter, we have all the Gaussians.
realign_iters="10 15"; 

numleaves=4200 
numgauss=20000 # Initial num-gauss.  Initializing states using tri2k model,
               # so can have a reasonably large number.
totgauss=40000 # Total num-gauss
incgauss=$[($totgauss-$numgauss)/$maxiterinc] # per-iter increment for #Gauss

silphonelist=`cat data/silphones.csl`

mkdir -p $dir

# Use all the SI-284 data. 

cp data/train.{scp,tra,utt2spk} $dir
cp $srcdir/tree $dir


# Split up the scp and related files to 3 parts; create spk2utt files.
scripts/split_scp.pl --utt2spk=$dir/train.utt2spk $dir/train.scp  $dir/train{1,2,3}.scp
scripts/split_scp.pl --utt2spk=$dir/train.utt2spk $dir/train.tra  $dir/train{1,2,3}.tra
scripts/split_scp.pl --utt2spk=$dir/train.utt2spk $dir/train{,1,2,3}.utt2spk
for n in 1 2 3 ""; do # The "" handles the un-split one.  Creating spk2utt files..
  scripts/utt2spk_to_spk2utt.pl $dir/train$n.utt2spk > $dir/train$n.spk2utt
done

# also see featspart below, used for sub-parts of the features.

# origfeats is feats with just LDA, no exponential trnsform.
# defaultfeats is for the "default" speaker-- used with alignment model.
origfeats="ark:splice-feats scp:$dir/train.scp ark:- | transform-feats $ldamat ark:- ark:- |"
# add s,cs to defaultfeats, as the program gmm-acc-stats-twofeats does random access on them.
defaultfeats="ark,s,cs:splice-feats scp:$dir/train.scp ark:- | transform-feats $defaultmat ark:- ark:- |"
feats="ark:splice-feats scp:$dir/train.scp ark:- | transform-feats $ldamat ark:- ark:- | transform-feats --utt2spk=ark:$dir/train.utt2spk \"ark:cat $dir/?.trans|\" ark:- ark:- |"
for n in 1 2 3; do
   featspart[$n]="ark,s,cs:splice-feats scp:$dir/train${n}.scp ark:- | transform-feats $ldamat ark:- ark:- | transform-feats --utt2spk=ark:$dir/train$n.utt2spk ark:$dir/$n.trans ark:- ark:- |"
   origfeatspart[$n]="ark,s,cs:splice-feats scp:$dir/train${n}.scp ark:- | transform-feats $ldamat ark:- ark:- |"
   defaultfeatspart[$n]="ark,s,cs:splice-feats scp:$dir/train${n}.scp ark:- | transform-feats $defaultmat ark:- ark:- |"
done

cp $srcdir/topo $dir


echo "Aligning all training data with alignment model"

rm -f $dir/.error
for n in 1 2 3; do
  compile-train-graphs $srcdir/tree $srcalimodel  data/L.fst ark:$dir/train${n}.tra ark:- 2>$dir/graphsold.${n}.log | \
   gmm-align-compiled $scale_opts --beam=8 --retry-beam=40 $srcalimodel ark:- "${defaultfeatspart[$n]}" "ark:|gzip -c > $dir/0.${n}.ali.gz" 2> $dir/align.0.${n}.log || touch $dir/.error &
done
wait;
[ -f $dir/.error ] &&  echo compile-graphs error RE old system && exit 1

echo "Computing ET transforms."
for n in 1 2 3; do
  ( ali-to-post "ark:gunzip -c $dir/0.${n}.ali.gz|" ark:- | \
   weight-silence-post 0.0 $silphonelist $srcalimodel ark:- ark:- | \
   gmm-post-to-gpost $srcalimodel "${defaultfeatspart[$n]}" ark,o:- ark:- | \
   gmm-est-et --spk2utt=ark:$dir/train$n.spk2utt $srcmodel $et "${origfeatspart[$n]}" ark,o:- \
     ark:$dir/$n.trans ark,t:$dir/$n.warp ) 2>$dir/est_et.$n.log || touch $dir/.error &
done
wait;
[ -f $dir/.error ] &&  echo Error computing ET transfirms && exit 1

# debug info for warping factors.
cat $dir/?.warp | scripts/process_warps.pl data/spk2gender.map > $dir/warps

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
    $dir/tree $dir/treeacc $dir/topo $dir/1.mdl \
    $srcdir/tree $srcmodel 2> $dir/init_model.log || exit 1;

# Mix down and mix up to get exactly the targets #Gauss
# (note: the tool does mix-down first regardless of option order.)
gmm-mixup --mix-down=$numgauss --mix-up=$numgauss $dir/1.mdl $dir/1.occs $dir/1.mdl \
    2>$dir/mixup.log || exit 1;

rm $dir/treeacc $dir/1.occs


# Convert alignments generated from monophone model, to use as initial alignments.

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
     [ -f $dir/.error ] &&  echo compile-graphs error && exit 1
   fi
   for n in 1 2 3; do
     gmm-acc-stats-ali --binary=false $dir/$x.mdl "${featspart[$n]}" \
       "ark:gunzip -c $dir/cur${n}.ali.gz|" $dir/$x.$n.acc 2> $dir/acc.$x.$n.log  || touch $dir/.error &
   done
   wait
   [ -f $dir/.error ] &&  echo accumulation error && exit 1
   gmm-sum-accs $dir/$x.acc $dir/$x.?.acc 2>$dir/sum_accs.$x.log || exit 1;
   rm $dir/$x.?.acc
   gmm-est --write-occs=$dir/$[$x+1].occs --mix-up=$numgauss $dir/$x.mdl $dir/$x.acc $dir/$[$x+1].mdl 2> $dir/update.$x.log || exit 1;
   rm $dir/$x.mdl $dir/$x.acc $dir/$x.occs 2>/dev/null
   if [ $x -le $maxiterinc ]; then 
      numgauss=$[$numgauss+$incgauss];
   fi
   x=$[$x+1];
done


( ali-to-post "ark:gunzip -c $dir/cur?.ali.gz|" ark:- | \
  gmm-acc-stats-twofeats $dir/$x.mdl "$feats" "$defaultfeats" ark:- $dir/$x.acc2 ) 2>$dir/acc_alimdl.log || exit 1;
  # Update model.
  gmm-est  --remove-low-count-gaussians=false $dir/$x.mdl $dir/$x.acc2 $dir/$x.alimdl \
      2>$dir/est_alimdl.log  || exit 1;
rm $dir/$x.acc2

( cd $dir; rm final.{mdl,alimdl} 2>/dev/null; 
  ln -s $x.mdl final.mdl; ln -s $x.alimdl final.alimdl )
 ln -s `pwd`/$et $dir/final.et 
