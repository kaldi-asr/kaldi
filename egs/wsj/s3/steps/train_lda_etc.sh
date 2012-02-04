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


# Triphone model training, on top of LDA + (possibly MLLT and possibly
# transforms).  This script does not train any transforms itself, but 
# obtains them (along with the mean normalization) from a supplied directory. 
# By default this is the
# "alignment directory" ($alidir) where the previous system's alignments
# are located, but you can give it the --transform-dir option to
# make it use transforms from a separate location than the alignment
# directory.  (e.g. useful if the transforms are obtained using
# get_transforms_from_ubm.sh)


nj=4
cmd=scripts/run.pl
transformdir=
stage=-3

for x in `seq 4`; do
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
  if [ $1 == "--transform-dir" ]; then
     transformdir=$2
     shift 2
  fi
  if [ $1 == "--stage" ]; then # stage to start training from, typically same as the iter you have a .mdl file;
     stage=$2                  # in case it failed part-way.  
     shift 2
  fi  
done

if [ $# != 6 ]; then
   echo "Usage: steps/train_lda_etc.sh <num-leaves> <tot-gauss> <data-dir> <lang-dir> <ali-dir> <exp-dir>"
   echo " e.g.: steps/train_lda_etc.sh 2500 15000 data/train_si84 data/lang exp/tri1_ali_si84 exp/tri2b"
   exit 1;
fi

if [ -f path.sh ]; then . path.sh; fi

numleaves=$1
totgauss=$2
data=$3
lang=$4
alidir=$5
dir=$6
[ -z $transformdir ] && transformdir=$alidir # Get transforms from $alidir if 
 # --transform-dir option not supplied.

if [ ! -f $alidir/final.mdl ]; then
  echo "Error: alignment dir $alidir does not contain final.mdl"
  exit 1;
fi

scale_opts="--transition-scale=1.0 --acoustic-scale=0.1 --self-loop-scale=0.1"
realign_iters="10 20 30";
oov_sym=`cat $lang/oov.txt`
silphonelist=`cat $lang/silphones.csl`
numiters=35    # Number of iterations of training
maxiterinc=25 # Last iter to increase #Gauss on.
numgauss=$numleaves
incgauss=$[($totgauss-$numgauss)/$maxiterinc] # per-iter increment for #Gauss

mkdir -p $dir/log

if [ ! -d $data/split$nj -o $data/split$nj -ot $data/feats.scp ]; then
  scripts/split_data.sh $data $nj
fi


n1=`get_splits.pl $nj | awk '{print $1}'`

[ -f $transformdir/$n1.trans ] && echo "Using speaker transforms from $transformdir" \
 && cp $transformdir/*.trans $dir # Just in case some other process needs to refer to them.

cp $transformdir/final.mat $dir || exit 1;

for n in `get_splits.pl $nj`; do
  featspart[$n]="ark,s,cs:apply-cmvn --norm-vars=false --utt2spk=ark:$data/split$nj/$n/utt2spk ark:$transformdir/$n.cmvn scp:$data/split$nj/$n/feats.scp ark:- | splice-feats ark:- ark:- | transform-feats $transformdir/final.mat ark:- ark:- |"
  if [ -f $transformdir/$n1.trans ]; then
    featspart[$n]="${featspart[$n]} transform-feats --utt2spk=ark:$data/split$nj/$n/utt2spk ark:$transformdir/$n.trans ark:- ark:- |"
  fi
done


if [ $stage -le -3 ]; then 
  # This stage assumes we won't need the context of silence, which
  # assumes something about $lang/roots.txt, but it seems pretty safe.
  echo "Accumulating tree stats"
  rm $dir/.error 2>/dev/null
  for n in `get_splits.pl $nj`; do
    $cmd $dir/log/acc_tree.$n.log \
    acc-tree-stats  --ci-phones=$silphonelist $alidir/final.mdl "${featspart[$n]}" \
      "ark:gunzip -c $alidir/$n.ali.gz|" $dir/$n.treeacc || touch $dir/.error &
  done
  wait
  [ -f $dir/.error ] && echo Error accumulating tree stats && exit 1;
  sum-tree-stats $dir/treeacc $dir/*.treeacc 2>$dir/log/sum_tree_acc.log || exit 1;
  rm $dir/*.treeacc

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

  gmm-init-model  --write-occs=$dir/1.occs  \
    $dir/tree $dir/treeacc $lang/topo $dir/1.mdl 2> $dir/log/init_model.log || exit 1;

  gmm-mixup --mix-up=$numgauss $dir/1.mdl $dir/1.occs $dir/1.mdl \
     2>$dir/log/mixup.log || exit 1;

  rm $dir/treeacc
fi



if [ $stage -le -2 ]; then
  # Convert alignments in $alidir, to use as initial alignments.
  # This assumes that $alidir was split in $nj pieces, just like the
  # current dir.

  echo "Converting old alignments"
  for n in `get_splits.pl $nj`; do # do this locally: it's very fast.
    convert-ali $alidir/final.mdl $dir/1.mdl $dir/tree \
     "ark:gunzip -c $alidir/$n.ali.gz|" "ark:|gzip -c >$dir/$n.ali.gz" \
      2>$dir/log/convert$n.log || exit 1;
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
fi

rm $dir/.error 2>/dev/null

x=1
while [ $x -lt $numiters ]; do
   if [ $stage -le $x ]; then 
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
     ## The main accumulation phase.. ##
     for n in `get_splits.pl $nj`; do 
       $cmd $dir/log/acc.$x.$n.log \
         gmm-acc-stats-ali  $dir/$x.mdl "${featspart[$n]}" \
           "ark,s,cs:gunzip -c $dir/$n.ali.gz|" $dir/$x.$n.acc || touch $dir/.error &
     done
     wait;
     [ -f $dir/.error ] && echo "Error accumulating stats on iteration $x" && exit 1;
     $cmd $dir/log/update.$x.log \
       gmm-est --write-occs=$dir/$[$x+1].occs --mix-up=$numgauss $dir/$x.mdl \
         "gmm-sum-accs - $dir/$x.*.acc |" $dir/$[$x+1].mdl || exit 1;
     rm $dir/$x.mdl $dir/$x.*.acc
  fi
  rm $dir/$x.occs 
  if [[ $x -le $maxiterinc ]]; then 
     numgauss=$[$numgauss+$incgauss];
  fi
  x=$[$x+1];
done

( cd $dir; rm final.mdl 2>/dev/null; ln -s $x.mdl final.mdl; ln -s $x.occs final.occs;
  ln -s `basename $cur_lda` final.mat )

echo Done
