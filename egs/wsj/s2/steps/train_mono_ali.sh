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
# Flat start and monophone training, with delta-delta features.
# This script applies cepstral mean normalization (per speaker).

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

if [ $# != 4 ]; then
   echo "Usage: steps/train_mono.sh <data-dir> <lang-dir> <ali-dir> <exp-dir>"
   echo " e.g.: steps/train_mono.sh data/train.1k data/lang exp/mono0a_ali exp/mono1a"
   exit 1;
fi


data=$1
lang=$2
alidir=$3
dir=$4

if [ -f path.sh ]; then . path.sh; fi

# Configuration:
scale_opts="--transition-scale=1.0 --acoustic-scale=0.1 --self-loop-scale=0.1"
numiters=35   # Number of iterations of training
maxiterinc=30 # Last iter to increase #Gauss on.
numgauss=300 # Initial num-Gauss (must be more than #states=3*phones).
states=146
totgauss=$[256*states] # Target #Gaussians.  
incgauss=$[($totgauss-$numgauss)/$maxiterinc] # per-iter increment for #Gauss
realign_iters="5 10 15 20 25 30";
oov_sym=`cat $lang/oov.txt`

mkdir -p $dir/log
if [ ! -d $data/split$nj -o $data/split$nj -ot $data/feats.scp ]; then
  split_data.sh $data $nj
fi

echo "Computing cepstral mean and variance statistics"

for n in `get_splits.pl $nj`; do # do this locally; it's fast.
  compute-cmvn-stats --spk2utt=ark:$data/split$nj/$n/spk2utt scp:$data/split$nj/$n/feats.scp \
    ark:$dir/$n.cmvn 2>$dir/log/cmvn$n.log || exit 1;
done

feats="ark:apply-cmvn --norm-vars=false --utt2spk=ark:$data/utt2spk \"ark:cat $dir/*.cmvn|\" scp:$data/feats.scp ark:- | add-deltas ark:- ark:- |"
for n in `get_splits.pl $nj`; do
  featspart[$n]="ark:apply-cmvn --norm-vars=false --utt2spk=ark:$data/split$nj/$n/utt2spk ark:$dir/$n.cmvn scp:$data/split$nj/$n/feats.scp ark:- | add-deltas ark:- ark:- |"
done


echo "Initializing monophone system."

if [ -f $lang/phonesets_mono.txt ]; then
  echo "Using shared phones from $lang/phonesets_mono.txt"
  # In recipes with stress and position markers, this pools together
  # the stats for the different versions of the same phone (also for 
  # the various silence phones).
  sym2int.pl $lang/phones.txt $lang/phonesets_mono.txt > $dir/phonesets.int
  shared_phones_opt="--shared-phones=$dir/phonesets.int"
fi

gmm-init-mono $shared_phones_opt "--train-feats=$feats subset-feats --n=10 ark:- ark:-|" $lang/topo 39  \
   $dir/1.mdl $dir/tree 2> $dir/log/init.log || exit 1;

rm $dir/.error 2>/dev/null

echo "Compiling training graphs"
for n in `get_splits.pl $nj`; do
  $cmd $dir/log/compile_graphs$n.log \
    compile-train-graphs $dir/tree $dir/1.mdl  $lang/L.fst \
      "ark:sym2int.pl --map-oov '$oov_sym' --ignore-first-field $lang/words.txt < $data/split$nj/$n/text|" \
      "ark:|gzip -c >$dir/$n.fsts.gz" || touch $dir/.error &
done
wait
[ -f $dir/.error ] && echo "Error compiling training graphs" && exit 1;

echo "Converting alignments"
for n in `get_splits.pl $nj`; do
  # Convert alignments generated from mono0a model, to use as initial alignments.
  convert-ali $alidir/final.mdl $dir/1.mdl $dir/tree "ark:gunzip -c $alidir/$n.ali.gz|" "ark:| gzip -c > $dir/$n.ali.gz" 2>$dir/convert.$n.log || touch $dir/.error &
done
wait
[ -f $dir/.error ] && echo "Alignment conversion" && exit 1;


beam=10
# note: using slightly wider beams for WSJ vs. RM.
x=1
while [ $x -lt $numiters ]; do
  echo "Pass $x"
  if echo $realign_iters | grep -w $x >/dev/null; then
    echo "Aligning data"
    for n in `get_splits.pl $nj`; do
     $cmd $dir/log/align.$x.$n.log \
      gmm-align-compiled $scale_opts --beam=$beam --retry-beam=$[$beam*4] $dir/$x.mdl \
        "ark:gunzip -c $dir/$n.fsts.gz|" "${featspart[$n]}" "ark,t:|gzip -c >$dir/$n.ali.gz" \
         || touch $dir/.error &
    done
    wait
    [ -f $dir/.error ] && echo "Error in pass $x alignment" && exit 1;
  fi
  for n in `get_splits.pl $nj`; do
    $cmd $dir/log/acc.$x.$n.log \
      gmm-acc-stats-ali  $dir/$x.mdl "${featspart[$n]}" "ark:gunzip -c $dir/$n.ali.gz|" \
        $dir/$x.$n.acc || touch $dir/.error &
  done
  wait
  [ -f $dir/.error ] && echo "Error in pass $n accumulation" && exit 1;
  $cmd $dir/log/update.$x.log \
    gmm-est --write-occs=$dir/$[$x+1].occs --mix-up=$numgauss $dir/$x.mdl \
      "gmm-sum-accs - $dir/$x.*.acc|" $dir/$[$x+1].mdl || exit 1;
  rm $dir/$x.mdl $dir/$x.*.acc $dir/$x.occs 2>/dev/null
  if [ $x -le $maxiterinc ]; then
     numgauss=$[$numgauss+$incgauss];
  fi
  x=$[$x+1]
done

( cd $dir; rm final.{mdl,occs} 2>/dev/null; ln -s $x.mdl final.mdl; ln -s $x.occs final.occs )

# Print out summary of the warning messages.
for x in $dir/log/*.log; do 
  n=`grep WARNING $x | wc -l`; 
  if [ $n -ne 0 ]; then echo $n warnings in $x; fi; 
done

echo Done

# example of showing the alignments:
# show-alignments data/lang/phones.txt $dir/30.mdl "ark:gunzip -c $dir/0.ali.gz|" | head -4

