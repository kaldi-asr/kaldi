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

if [ $# != 3 ]; then
   echo "Usage: steps/train_mono.sh <data-dir> <lang-dir> <exp-dir>"
   echo " e.g.: steps/train_mono.sh data/train.1k data/lang exp/mono"
   exit 1;
fi


data=$1
lang=$2
dir=$3

if [ -f path.sh ]; then . path.sh; fi

# Configuration:
scale_opts="--transition-scale=1.0 --acoustic-scale=0.1 --self-loop-scale=0.1"
numiters=40    # Number of iterations of training
maxiterinc=30 # Last iter to increase #Gauss on.
numgauss=300 # Initial num-Gauss (must be more than #states=3*phones).
totgauss=1000 # Target #Gaussians.  
incgauss=$[($totgauss-$numgauss)/$maxiterinc] # per-iter increment for #Gauss
realign_iters="1 2 3 4 5 6 7 8 9 10 12 14 16 18 20 23 26 29 32 35 38";
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
   $dir/0.mdl $dir/tree 2> $dir/log/init.log || exit 1;

rm $dir/.error 2>/dev/null

echo "Compiling training graphs"
for n in `get_splits.pl $nj`; do
  $cmd $dir/log/compile_graphs$n.log \
    compile-train-graphs $dir/tree $dir/0.mdl  $lang/L.fst \
      "ark:sym2int.pl --map-oov '$oov_sym' --ignore-first-field $lang/words.txt < $data/split$nj/$n/text|" \
      "ark:|gzip -c >$dir/$n.fsts.gz" || touch $dir/.error &
done
wait
[ -f $dir/.error ] && echo "Error compiling training graphs" && exit 1;

echo "Aligning data equally (pass 0)"

for n in `get_splits.pl $nj`; do
  $cmd $dir/log/align.0.$n.log \
    align-equal-compiled "ark:gunzip -c $dir/$n.fsts.gz|" "${featspart[$n]}" ark,t,f:-  \| \
      gmm-acc-stats-ali --binary=true $dir/0.mdl "${featspart[$n]}" ark:- \
        $dir/0.$n.acc || touch $dir/.error &
done
wait
[ -f $dir/.error ] && echo "Error in pass 0 accumulation" && exit 1;

# In the following steps, the --min-gaussian-occupancy=3 option is important, otherwise
# we fail to est "rare" phones and later on, they never align properly.

gmm-est --min-gaussian-occupancy=3  --mix-up=$numgauss \
    $dir/0.mdl "gmm-sum-accs - $dir/0.*.acc|" $dir/1.mdl 2> $dir/log/update.0.log || exit 1;

rm $dir/0.*.acc

beam=6 # will change to 10 below after 1st pass
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
      gmm-acc-stats-ali --binary=false $dir/$x.mdl "${featspart[$n]}" "ark:gunzip -c $dir/$n.ali.gz|" \
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
  beam=10
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

