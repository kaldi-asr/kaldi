#!/bin/bash
# Copyright 2010-2011 Microsoft Corporation

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


# Train UBM from a trained HMM/GMM system [with splice+LDA+[MLLT/ET/MLLT+SAT] features]
# Alignment directory is used for the CMN and transforms.

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

if [ $# != 5 ]; then
  echo "Usage: steps/train_ubm_lda_etc.sh <num-comps> <data-dir> <lang-dir> <ali-dir> <exp-dir>"
  echo " e.g.: steps/train_ubm_lda_etc.sh 400 data/train_si84 data/lang exp/tri2b_ali_si84 exp/ubm3c"
  exit 1;
fi

if [ -f path.sh ]; then . path.sh; fi

numcomps=$1
data=$2
lang=$3
alidir=$4
dir=$5

mkdir -p $dir/log

if [ ! -f $data/split$nj -o $data/split$nj -ot $data/feats.scp ]; then
  scripts/split_data.sh $data $nj
fi

n1=`get_splits.pl $nj | awk '{print $1}'`
[ -f $alidir/$n1.trans ] && echo "Using speaker transforms from $alidir"

for n in `get_splits.pl $nj`; do
  featspart[$n]="ark:apply-cmvn --norm-vars=false --utt2spk=ark:$data/split$nj/$n/utt2spk ark:$alidir/$n.cmvn scp:$data/split$nj/$n/feats.scp ark:- | splice-feats ark:- ark:- | transform-feats $alidir/final.mat ark:- ark:- |"
  if [ -f $alidir/$n1.trans ]; then
    featspart[$n]="${featspart[$n]} transform-feats --utt2spk=ark:$data/split$nj/$n/utt2spk ark:$alidir/$n.trans ark:- ark:- |"
  fi
done

intermediate=2000
if [ $[$numcomps*2] -gt $intermediate ]; then
  intermediate=$[$numcomps*2];
fi

echo "Clustering model $alidir/final.mdl to get initial UBM"
# typically: --intermediate-numcomps=2000 --ubm-numcomps=400
init-ubm --intermediate-numcomps=$intermediate --ubm-numcomps=$numcomps --verbose=2 \
   --fullcov-ubm=true $alidir/final.mdl $alidir/final.occs \
    $dir/0.ubm 2> $dir/log/cluster.log

rm $dir/.error 2>/dev/null
for x in 0 1 2 3; do
  echo "Pass $x"
  for n in `get_splits.pl $nj`; do
    $cmd $dir/log/acc.$x.$n.log \
      fgmm-global-acc-stats --diag-gmm-nbest=15 --binary=false --verbose=2 $dir/$x.ubm "${featspart[$n]}" \
        $dir/$x.$n.acc || touch $dir/.error &
  done
  wait
  [ -f $dir/.error ] && echo "Error accumulating stats for UBM estimation on pass $x" && exit 1;
  $cmd $dir/log/update.$x.log \
    fgmm-global-est --verbose=2 $dir/$x.ubm "fgmm-global-sum-accs - $dir/$x.*.acc |" \
      $dir/$[$x+1].ubm || exit 1;
  rm $dir/$x.*.acc $dir/$x.ubm
done

mv $dir/4.ubm $dir/final.ubm || exit 1;

