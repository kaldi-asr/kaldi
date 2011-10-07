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

# Trains SGMM on top of LDA plus [something] features, where the [something]
# might be e.g. MLLT, or some kind of speaker-specific transform.

if [ $# != 9 ]; then
   echo "Usage: steps/train_sgmm_lda_etc.sh <num-leaves> <num-substates> <phone-dim> <spk-dim> <data-dir> <lang-dir> <ali-dir> <ubm> <exp-dir>"
   echo " e.g.: steps/train_sgmm_lda_etc.sh 3500 10000 41 40 data/train_si84 data/lang exp/tri2b_ali_si84 exp/ubm3c/final.ubm exp/sgmm3c"
   exit 1;
fi

if [ -f path.sh ]; then . path.sh; fi

# This is SGMM with speaker vectors, on top of LDA+[something] features.
# Any speaker-specific transforms are obtained from the alignment directory.
# To be run from ..

numleaves=$1
totsubstates=$2
phndim=$3
spkdim=$4
data=$5
lang=$6
alidir=$7
ubm=$8
dir=$9

mkdir -p $dir || exit 1;
cp $alidir/final.mat $dir/final.mat || exit 1;

scale_opts="--transition-scale=1.0 --acoustic-scale=0.1 --self-loop-scale=0.1"
oov_sym=`cat $lang/oov.txt`

numiters=25   # Total number of iterations
numiters_alimdl=3 # Number of iterations for estimating alignment model.
maxiterinc=15 # Last iter to increase #substates on.
realign_iters="5 10 15"; 
spkvec_iters="5 8 12 17"
add_dim_iters="6 8 10 12"; # Iters on which to increase phn dim and/or spk dim,
   # if necessary, In most cases, either none of these or only the first of these 
   # will have any effect (we increase in increments of [feature dim])

silphonelist=`cat $lang/silphones.csl`


numsubstates=$numleaves # Initial #-substates.
incsubstates=$[($totsubstates-$numsubstates)/$maxiterinc] # per-iter increment for #substates


for n in `get_splits.pl $nj`; do
  # Initially don't have speaker vectors, but change this after we estimate them.
  spkvecs_opt[$n]=
  gselect_opt[$n]="--gselect=ark,s,cs:gunzip -c $dir/$n.gselect.gz|"
done
randprune=0.1
mkdir -p $dir/log 

feats="ark,s,cs:apply-cmvn --norm-vars=false --utt2spk=ark:$data/utt2spk 'ark:cat $alidir/*.cmvn|' scp:$data/feats.scp ark:- | splice-feats ark:- ark:- | transform-feats $dir/final.mat ark:- ark:- |"
n1=`get_splits.pl $nj | awk '{print $1}'`
if [ -f $alidir/$n1.trans ]; then
  echo "Using speaker transforms from $alidir"
  feats="$feats transform-feats --utt2spk=ark:$data/utt2spk 'ark:cat $alidir/*.trans|' ark:- ark:- |"
fi
for n in `get_splits.pl $nj`; do
  featspart[$n]="ark,s,cs:apply-cmvn --norm-vars=false --utt2spk=ark:$data/split$nj/$n/utt2spk ark:$alidir/$n.cmvn scp:$data/split$nj/$n/feats.scp ark:- | splice-feats ark:- ark:- | transform-feats $dir/final.mat ark:- ark:- |"
  if [ -f $alidir/0.trans ]; then
    featspart[$n]="${featspart[$n]} transform-feats --utt2spk=ark:$data/split$nj/$n/utt2spk ark:$alidir/$n.trans ark:- ark:- |"
  fi
done


if [ ! -f $ubm ]; then
  echo "No UBM in $ubm"
  exit 1;
fi

# Get stats to build the tree.
echo "Accumulating tree stats"
$cmd $dir/log/acc_tree.log \
 acc-tree-stats  --ci-phones=$silphonelist $alidir/final.mdl "$feats" "ark:gunzip -c $alidir/*ali.gz|" \
  $dir/treeacc || exit 1;


echo "Computing questions for tree clustering"
# preparing questions, roots file...
sym2int.pl $lang/phones.txt $lang/phonesets_cluster.txt > $dir/phonesets.txt || exit 1;
cluster-phones $dir/treeacc $dir/phonesets.txt $dir/questions.txt 2> $dir/log/questions.log || exit 1;
sym2int.pl $lang/phones.txt $lang/extra_questions.txt >> $dir/questions.txt
compile-questions $lang/topo $dir/questions.txt $dir/questions.qst 2>$dir/log/compile_questions.log || exit 1;
sym2int.pl --ignore-oov $lang/phones.txt $lang/roots.txt > $dir/roots.txt

echo "Building tree"
$cmd $dir/log/train_tree.log \
  build-tree --verbose=1 --max-leaves=$numleaves \
    $dir/treeacc $dir/roots.txt \
    $dir/questions.qst $lang/topo $dir/tree || exit 1;


# the sgmm-init program accepts a GMM, so we just create a temporary GMM "0.gmm"

$cmd $dir/log/init_gmm.log \
  gmm-init-model  --write-occs=$dir/0.occs  \
    $dir/tree $dir/treeacc $lang/topo $dir/0.gmm || exit 1;


# The next line is a bit of a hack to work out the feature dim.  The program
# feat-to-len returns the #rows of each matrix, which for the transform matrix,
# is the feature dim.
featdim=`feat-to-len "scp:echo foo $alidir/final.mat|" ark,t:- 2>/dev/null | awk '{print $2}'`



# Note: if phndim and/or spkdim are higher than you can initialize with,
# sgmm-init will just make them as high as it can (later we'll increase)

$cmd $dir/log/init_sgmm.log \
  sgmm-init --phn-space-dim=$phndim --spk-space-dim=$spkdim $lang/topo $dir/tree $ubm \
    $dir/0.mdl || exit 1;

rm $dir/.error 2>/dev/null
echo "Doing Gaussian selection"
for n in `get_splits.pl $nj`; do
  $cmd $dir/log/gselect$n.log \
    sgmm-gselect $dir/0.mdl "${featspart[$n]}" "ark,t:|gzip -c > $dir/$n.gselect.gz" \
   || touch $dir/.error &
done
wait;
[ -f $dir/.error ] && echo "Error doing Gaussian selection" && exit 1;

echo "Converting alignments"  # don't bother parallelizing; very fast.
for n in `get_splits.pl $nj`; do
  convert-ali $alidir/final.mdl $dir/0.mdl $dir/tree "ark:gunzip -c $alidir/$n.ali.gz|" \
     "ark:|gzip -c >$dir/$n.ali.gz" 2>$dir/log/convert$n.log 
done

echo "Compiling training graphs"
for n in `get_splits.pl $nj`; do
  $cmd $dir/log/compile_graphs$n.log \
    compile-train-graphs $dir/tree $dir/0.mdl  $lang/L.fst  \
     "ark:sym2int.pl --map-oov \"$oov_sym\" --ignore-first-field $lang/words.txt < $data/split$nj/$n/text |" \
     "ark:|gzip -c >$dir/$n.fsts.gz" || touch $dir/.error &
done
wait;
[ -f $dir/.error ] && echo "Error compiling training graphs" && exit 1;

x=0
while [ $x -lt $numiters ]; do
   echo "Pass $x ... "
   if echo $realign_iters | grep -w $x >/dev/null; then
      echo "Aligning data"
      for n in `get_splits.pl $nj`; do
        $cmd $dir/log/align.$x.$n.log  \
          sgmm-align-compiled ${spkvecs_opt[$n]} $scale_opts "${gselect_opt[$n]}" \
             --utt2spk=ark:$data/split$nj/$n/utt2spk --beam=8 --retry-beam=40 \
             $dir/$x.mdl "ark:gunzip -c $dir/$n.fsts.gz|" "${featspart[$n]}" \
             "ark:|gzip -c >$dir/$n.ali.gz" || touch $dir/.error &
      done
      wait;
      [ -f $dir/.error ] && echo "Error realigning data on iter $x" && exit 1;
   fi
   if [ $spkdim -gt 0 ] && echo $spkvec_iters | grep -w $x >/dev/null; then
     for n in `get_splits.pl $nj`; do
       $cmd $dir/log/spkvecs.$x.$n.log \
         ali-to-post "ark:gunzip -c $dir/$n.ali.gz|" ark:- \| \
           weight-silence-post 0.01 $silphonelist $dir/$x.mdl ark:- ark:- \| \
           sgmm-est-spkvecs --spk2utt=ark:$data/split$nj/$n/spk2utt \
             ${spkvecs_opt[$n]} "${gselect_opt[$n]}" \
             --rand-prune=$randprune $dir/$x.mdl \
          "${featspart[$n]}" ark,s,cs:- ark:$dir/tmp$n.vecs  \
         && mv $dir/tmp$n.vecs $dir/$n.vecs || touch $dir/.error &
       spkvecs_opt[$n]="--spk-vecs=ark:$dir/$n.vecs"
     done
     wait;
     [ -f $dir/.error ] && echo "Error computing speaker vectors on iter $x" && exit 1;     
   fi  
   if [ $x -eq 0 ]; then
     flags=vwcS # On first iter, don't update M or N.
   elif [ $spkdim -gt 0 -a $[$x%2] -eq 1 -a $x -ge `echo $spkvec_iters | awk '{print $1}'` ]; then 
     # Update N if we have spk-space and x is even, and we're at least at 1st spkvec iter.
     flags=vNwcS
   else # Else update M but not N.
     flags=vMwcS
   fi

   for n in `get_splits.pl $nj`; do
     $cmd $dir/log/acc.$x.$n.log \
       sgmm-acc-stats ${spkvecs_opt[$n]} --utt2spk=ark:$data/split$nj/$n/utt2spk \
         --update-flags=$flags "${gselect_opt[$n]}" --rand-prune=$randprune \
         $dir/$x.mdl "${featspart[$n]}" "ark:ali-to-post 'ark:gunzip -c $dir/$n.ali.gz|' ark:-|" \
         $dir/$x.$n.acc || touch $dir/.error &
   done
   wait;
   [ -f $dir/.error ] && echo "Error accumulating stats on iter $x" && exit 1;     

   add_dim_opts=
   if echo $add_dim_iters | grep -w $x >/dev/null; then
     add_dim_opts="--increase-phn-dim=$phndim --increase-spk-dim=$spkdim"
   fi

   $cmd $dir/log/update.$x.log \
     sgmm-est --update-flags=$flags --split-substates=$numsubstates $add_dim_opts \
       --write-occs=$dir/$[$x+1].occs $dir/$x.mdl "sgmm-sum-accs - $dir/$x.*.acc|" \
     $dir/$[$x+1].mdl || exit 1;

   rm $dir/$x.mdl $dir/$x.*.acc
   rm $dir/$x.occs 
   if [ $x -lt $maxiterinc ]; then
     numsubstates=$[$numsubstates+$incsubstates]
   fi
   x=$[$x+1];
done

( cd $dir; rm final.mdl final.occs 2>/dev/null; 
  ln -s $x.mdl final.mdl; 
  ln -s $x.occs final.occs )

if [ $spkdim -gt 0 ]; then
  # If we have speaker vectors, we need an alignment model.
  # The point of this last phase of accumulation is to get Gaussian-level
  # alignments with the speaker vectors but accumulate stats without
  # any speaker vectors; we re-estimate M, w, c and S to get a model
  # that's compatible with not having speaker vectors.

  # We do this for a few iters, in this recipe.
  cur_alimdl=$dir/$x.mdl
  y=0;
  while [ $y -lt $numiters_alimdl ]; do
    echo "Pass $y of building alignment model"
    if [ $y -eq 0 ]; then
      flags=MwcS # First time don't update v...
    else
      flags=vMwcS
    fi
    for n in `get_splits.pl $nj`; do
      $cmd $dir/log/acc_ali.$y.$n.log \
        ali-to-post "ark:gunzip -c $dir/$n.ali.gz|" ark:- \| \
          sgmm-post-to-gpost ${spkvecs_opt[$n]} "${gselect_opt[$n]}" \
            --utt2spk=ark:$data/split$nj/$n/utt2spk $dir/$x.mdl "${featspart[$n]}" ark,s,cs:- ark:- \| \
          sgmm-acc-stats-gpost --update-flags=$flags  $cur_alimdl "${featspart[$n]}" \
            ark,s,cs:- $dir/$y.$n.aliacc || touch $dir/.error &
    done
    wait;
    [ -f $dir/.error ] && echo "Error accumulating stats for alignment model on iter $y" && exit 1;
    $cmd $dir/log/update_ali.$y.log \
       sgmm-est --update-flags=$flags --remove-speaker-space=true $cur_alimdl \
       "sgmm-sum-accs - $dir/$y.*.aliacc|" $dir/$[$y+1].alimdl || exit 1;
    # [ $y -gt 0 ]  && rm $dir/$y.alimdl
    cur_alimdl=$dir/$[$y+1].alimdl
    y=$[$y+1]
  done
  (cd $dir; rm final.alimdl 2>/dev/null; ln -s $y.alimdl final.alimdl )
fi


# Print out summary of the warning messages.
for x in $dir/log/*.log; do 
  n=`grep WARNING $x | wc -l`; 
  if [ $n -ne 0 ]; then echo $n warnings in $x; fi; 
done

echo Done
