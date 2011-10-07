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
# Triphone model training, using cepstral mean normalization, plus
# splice-9-frames and LDA, plus MLLT/STC. 

nj=4
cmd=scripts/run.pl
for x in 1 2; do
  if [ $1 == "--num-jobs" ]; then
     shift
     nj=$1
     shift
  fi
  if [ $1 == "--cmd" ]; then
     shift
     cmd=$1
     shift
  fi  
done

if [ $# != 6 ]; then
   echo "Usage: steps/train_lda_et.sh <num-leaves> <tot-gauss> <data-dir> <lang-dir> <ali-dir> <exp-dir>"
   echo " e.g.: steps/train_lda_et.sh 2500 15000 data/train_si84 data/lang exp/tri1_ali_si84 exp/tri2c"
   exit 1;
fi

if [ -f path.sh ]; then . path.sh; fi

numleaves=$1
totgauss=$2
data=$3
lang=$4
alidir=$5
dir=$6

if [ ! -f $alidir/final.mdl ]; then
  echo "Error: alignment dir $alidir does not contain final.mdl"
  exit 1;
fi

scale_opts="--transition-scale=1.0 --acoustic-scale=0.1 --self-loop-scale=0.1"
realign_iters="10 20 30";
numiters_et=15
normtype=offset # et option; could be offset [recommended], or none
oov_sym="<SPOKEN_NOISE>" # Map OOVs to this in training.
grep SPOKEN_NOISE $lang/words.txt >/dev/null || echo "Warning: SPOKEN_NOISE not in dictionary"
silphonelist=`cat $lang/silphones.csl`
numiters=35    # Number of iterations of training
maxiterinc=25 # Last iter to increase #Gauss on.
numgauss=$numleaves
incgauss=$[($totgauss-$numgauss)/$maxiterinc] # per-iter increment for #Gauss
randprune=4.0

mkdir -p $dir/log $dir/warps

if [ ! -f $data/split$nj -o $data/split$nj -ot $data/feats.scp ]; then
  scripts/split_data.sh $data $nj
fi

# basefeats is all the feats, transformed with lda.mat-- just needed for tree accumulation.
basefeats="ark:apply-cmvn --norm-vars=false --utt2spk=ark:$data/utt2spk \"ark:cat $alidir/*.cmvn|\" scp:$data/feats.scp ark:- | splice-feats ark:- ark:- | transform-feats $dir/lda.mat ark:- ark:- |"

for n in `get_splits.pl $nj`; do
  splicedfeatspart[$n]="ark:apply-cmvn --norm-vars=false --utt2spk=ark:$data/split$nj/$n/utt2spk ark:$alidir/$n.cmvn scp:$data/split$nj/$n/feats.scp ark:- | splice-feats ark:- ark:- |"
  basefeatspart[$n]="${splicedfeatspart[$n]} transform-feats $dir/lda.mat ark:- ark:- |"
  featspart[$n]="${basefeatspart[$n]}" # This gets overwritten later in the script.
done

echo "Accumulating LDA statistics."

rm $dir/.error 2>/dev/null

for n in `get_splits.pl $nj`; do
  $cmd $dir/log/lda_acc.$n.log \
    ali-to-post "ark:gunzip -c $alidir/$n.ali.gz|" ark:- \| \
      weight-silence-post 0.0 $silphonelist $alidir/final.mdl ark:- ark:- \| \
      acc-lda --rand-prune=$randprune $alidir/final.mdl "${splicedfeatspart[$n]}" ark,s,cs:- \
       $dir/lda.$n.acc || touch $dir/.error &
done
wait
[ -f $dir/.error ] && echo "Error accumulating LDA stats" && exit 1;
est-lda $dir/lda.mat $dir/lda.*.acc 2>$dir/log/lda_est.log || exit 1; # defaults to dim=40
rm $dir/lda.*.acc
cur_lda=$dir/0.mat

# The next stage assumes we won't need the context of silence, which
# assumes something about $lang/roots.txt, but it seems pretty safe.
echo "Accumulating tree stats"
$cmd $dir/log/acc_tree.log \
  acc-tree-stats  --ci-phones=$silphonelist $alidir/final.mdl "$basefeats" \
    "ark:gunzip -c $alidir/*.ali.gz|" $dir/treeacc || exit 1;

echo "Computing questions for tree clustering"
# preparing questions, roots file...
scripts/sym2int.pl $lang/phones.txt $lang/phonesets_cluster.txt > $dir/phonesets.txt || exit 1;
cluster-phones $dir/treeacc $dir/phonesets.txt $dir/questions.txt 2> $dir/log/questions.log || exit 1;
scripts/sym2int.pl $lang/phones.txt $lang/extra_questions.txt >> $dir/questions.txt
compile-questions $lang/topo $dir/questions.txt $dir/questions.qst 2>$dir/log/compile_questions.log || exit 1;
scripts/sym2int.pl --ignore-oov $lang/phones.txt $lang/roots.txt > $dir/roots.txt

echo "Building tree"
$cmd $dir/log/train_tree.log \
  build-tree --verbose=1 --max-leaves=$numleaves \
    $dir/treeacc $dir/roots.txt \
    $dir/questions.qst $lang/topo $dir/tree || exit 1;

$cmd $dir/log/init_model.log \
  gmm-init-model  --write-occs=$dir/1.occs  \
    $dir/tree $dir/treeacc $lang/topo $dir/1.mdl || exit 1;

$cmd $dir/log/mixup.log \
  gmm-mixup --mix-up=$numgauss $dir/1.mdl $dir/1.occs $dir/1.mdl || exit 1;

$cmd $dir/log/init_et.log \
  gmm-init-et --normalize-type=$normtype --binary=false --dim=40 $dir/1.et || exit 1

rm $dir/treeacc

# Convert alignments in $alidir, to use as initial alignments.
# This assumes that $alidir was split in $nj pieces, just like the
# current dir.

echo "Converting old alignments"  # do this locally; it's fast.
for n in `get_splits.pl $nj`; do
  convert-ali $alidir/final.mdl $dir/1.mdl $dir/tree \
   "ark:gunzip -c $alidir/$n.ali.gz|" "ark:|gzip -c >$dir/$n.ali.gz" \
    2>$dir/log/convert$n.log  || exit 1;
done

# Make training graphs (this is split in $nj parts).
echo "Compiling training graphs"
rm $dir/.error 2>/dev/null
for n in `get_splits.pl $nj`; do
  $cmd $dir/log/compile_graphs$n.log \
    compile-train-graphs $dir/tree $dir/1.mdl  $lang/L.fst  \
      "ark:scripts/sym2int.pl --map-oov \"$oov_sym\" --ignore-first-field $lang/words.txt < $data/split$nj/$n/text |" \
      "ark:|gzip -c >$dir/$n.fsts.gz" || touch $dir/.error &
done
wait;
[ -f $dir/.error ] && echo "Error compiling training graphs" && exit 1;

x=1
while [ $x -lt $numiters ]; do
  echo Pass $x
  if echo $realign_iters | grep -w $x >/dev/null; then
    echo "Aligning data"
    for n in `get_splits.pl $nj`; do
      $cmd $dir/log/align.$x.$n.log \
        gmm-align-compiled $scale_opts --beam=10 --retry-beam=40 $dir/$x.mdl \
          "ark:gunzip -c $dir/$n.fsts.gz|" "${featspart[$n]}" \
          "ark:|gzip -c >$dir/$n.ali.gz" || touch $dir/.error &
    done
    wait;
    [ -f $dir/.error ] && echo "Error aligning data on iteration $x" && exit 1;
  fi

  if [ $x -lt $numiters_et ]; then
    echo "Re-estimating ET transforms"
    for n in `get_splits.pl $nj`; do
      $cmd $dir/log/trans.$x.$n.log \
        ali-to-post "ark:gunzip -c $dir/$n.ali.gz|" ark:- \| \
        weight-silence-post 0.0 $silphonelist $dir/$x.mdl ark:- ark:- \| \
        rand-prune-post $randprune ark:- ark:- \| \
        gmm-post-to-gpost $dir/$x.mdl "${featspart[$n]}" ark:- ark:- \| \
        gmm-est-et --spk2utt=ark:$data/split$nj/$n/spk2utt $dir/$x.mdl $dir/$x.et "${basefeatspart[$n]}" \
          ark,s,cs:- ark:$dir/$n.trans.tmp ark,t:$dir/warps/$x.$n.warp || touch $dir/.error &
    done
    wait
    [ -f $dir/.error ] && echo "Error computing ET transforms on iteration $x" && exit 1;
    for n in `get_splits.pl $nj`; do 
      mv $dir/$n.trans.tmp $dir/$n.trans || exit 1;
      featspart[$n]="${basefeatspart[$n]} transform-feats --utt2spk=ark:$data/split$nj/$n/utt2spk ark:$dir/$n.trans ark:- ark:- |"
    done
  fi

  for n in `get_splits.pl $nj`; do
    $cmd $dir/log/acc.$x.$n.log \
      gmm-acc-stats-ali --binary=false $dir/$x.mdl "${featspart[$n]}" \
        "ark:gunzip -c $dir/$n.ali.gz|" $dir/$x.$n.acc || touch $dir/.error &
  done
  wait;
  [ -f $dir/.error ] && echo "Error accumulating stats on iteration $x" && exit 1;
  $cmd $dir/log/update.$x.log  \
    gmm-est --write-occs=$dir/$[$x+1].occs --mix-up=$numgauss $dir/$x.mdl \
      "gmm-sum-accs - $dir/$x.*.acc |" $dir/$[$x+1].mdl || exit 1;
  rm $dir/$x.mdl $dir/$x.*.acc
  rm $dir/$x.occs 

  x1=$[$x+1];
  if [ $x -lt $numiters_et ]; then  
    # Alternately estimate either A or B.
    if [ $[$x%2] == 0 ]; then  # Estimate A:
      for n in `get_splits.pl $nj`; do
        $cmd $dir/log/acc_a.$x.$n.log \
          ali-to-post "ark:gunzip -c $dir/$n.ali.gz|" ark:- \| \
          weight-silence-post 0.0 $silphonelist $dir/$x1.mdl ark:- ark:- \| \
          rand-prune-post $randprune ark:- ark:- \| \
          gmm-post-to-gpost $dir/$x1.mdl "${featspart[$n]}" ark:- ark:- \| \
          gmm-et-acc-a --spk2utt=ark:$data/split$nj/$n/spk2utt --verbose=1 $dir/$x1.mdl $dir/$x.et "${basefeatspart[$n]}" \
            ark,s,cs:- $dir/$x.$n.et_acc_a || touch $dir/.error &
      done
      wait
      [ -f $dir/.error ] && echo "Error accumulating ET stats for A on iter $x" && exit 1;
      gmm-et-est-a --verbose=1 $dir/$x.et $dir/$x1.et $dir/$x.*.et_acc_a 2> $dir/log/update_a.$x.log || exit 1;
      rm $dir/$x.*.et_acc_a
    else
      for n in `get_splits.pl $nj`; do
        $cmd $dir/log/acc_b.$x.$n.log \
          ali-to-post "ark:gunzip -c $dir/$n.ali.gz|" ark:- \| \
          weight-silence-post 0.0 $silphonelist $dir/$x1.mdl ark:- ark:- \| \
          gmm-acc-mllt --rand-prune=$randprune $dir/$x1.mdl "${featspart[$n]}" ark:- \
            $dir/$x.$n.mllt_acc || touch $dir/.error &
      done
      wait
      [ -f $dir/.error ] && echo "Error accumulating ET stats for A on iter $x" && exit 1;
      est-mllt $dir/$x.mat $dir/$x.*.mllt_acc 2> $dir/log/update_b.$x.log || exit 1;
      gmm-et-apply-c $dir/$x.et $dir/$x.mat $dir/$x1.et 2>>$dir/log/update_b.$x.log || exit 1;
      gmm-transform-means $dir/$x.mat $dir/$x1.mdl $dir/$x1.mdl 2>> $dir/log/update_b.$x.log || exit 1;
      # Modify current transforms by premultiplying by C.
      for n in `get_splits.pl $nj`; do
        compose-transforms $dir/$x.mat ark:$dir/$n.trans ark:$dir/tmp.trans 2>> $dir/update_b.$x.log || exit 1;
        mv $dir/tmp.trans $dir/$n.trans
      done
      rm $dir/$x.mat
      rm $dir/$x.*.mllt_acc
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

for n in `get_splits.pl $nj`; do
  defaultfeatspart[$n]="${basefeatspart[$n]} transform-feats $dir/B.mat ark:- ark:- |"
done

# Accumulate stats for "alignment model" which is as the model but with
# the default features (shares Gaussian-level alignments).
for n in `get_splits.pl $nj`; do
  $cmd $dir/log/acc_alimdl.$n.log \
    ali-to-post "ark:gunzip -c $dir/$n.ali.gz|" ark:-  \| \
    gmm-acc-stats-twofeats $dir/$x.mdl "${featspart[$n]}" "${defaultfeatspart[$n]}" \
      ark:- $dir/$x.$n.acc2 || touch $dir/.error &
done
wait;
[ -f $dir/.error ] && echo "Error accumulating alignment statistics." && exit 1;
# Update model.
$cmd $dir/log/est_alimdl.log \
  gmm-est --write-occs=$dir/final.occs --remove-low-count-gaussians=false $dir/$x.mdl \
    "gmm-sum-accs - $dir/$x.*.acc2|" $dir/$x.alimdl || exit 1;
rm $dir/$x.*.acc2

# The following files may be useful for display purposes.
for y in 2 3 4 5 6 7 8 9 10 11 12 13 14; do
  cat $dir/warps/$y.*.warp | scripts/process_warps.pl $data/spk2gender > $dir/warps/$y.warp_info
done

( cd $dir; 
  ln -s $x.mdl final.mdl; 
  ln -s $x.alimdl final.alimdl )
# we already have final.mat and final.occs and final.et

echo Done
