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


# Train UBM from a trained HMM/GMM system [with splice+LDA+MLLT features]

if [ $# != 4 ]; then
   echo "Usage: steps/train_ubm_lda_mllt.sh <data-dir> <lang-dir> <ali-dir> <exp-dir>"
   echo " e.g.: steps/train_ubm_lda_mllt.sh data/train data/lang exp/tri2b_ali exp/ubm2d"
   exit 1;
fi

if [ -f path.sh ]; then . path.sh; fi

data=$1
lang=$2
alidir=$3
dir=$4

mkdir -p $dir
mat=$alidir/final.mat


feats="ark:apply-cmvn --norm-vars=false --utt2spk=ark:$data/utt2spk ark:$alidir/cmvn.ark scp:$data/feats.scp ark:- | splice-feats ark:- ark:- | transform-feats $alidir/final.mat ark:- ark:- |"

init-ubm --intermediate-numcomps=2000 --ubm-numcomps=400 --verbose=2 \
    --fullcov-ubm=true $alidir/final.mdl $alidir/final.occs \
    $dir/0.ubm 2> $dir/cluster.log


for x in 0 1 2 3; do
    echo "Pass $x"
    fgmm-global-acc-stats --diag-gmm-nbest=15 --binary=false --verbose=2 $dir/$x.ubm "$feats" $dir/$x.acc \
	2> $dir/acc.$x.log  || exit 1;
    fgmm-global-est --verbose=2 $dir/$x.ubm $dir/$x.acc \
	$dir/$[$x+1].ubm 2> $dir/update.$x.log || exit 1;
    rm $dir/$x.acc $dir/$x.ubm
done

mv $dir/$x.ubm $dir/final.ubm


