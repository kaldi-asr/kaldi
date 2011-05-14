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


# Train UBM from a trained HMM/GMM system.

if [ -f path.sh ]; then . path.sh; fi

dir=exp/ubma
mkdir -p $dir
srcdir=exp/tri1

init-ubm --intermediate-numcomps=2000 --ubm-numcomps=400 --verbose=2 \
    --fullcov-ubm=true $srcdir/final.mdl $srcdir/final.occs \
    $dir/0.ubm 2> $dir/cluster.log



subset[0]=1000
subset[1]=1500
subset[2]=2000
subset[3]=2500

for x in 0 1 2 3; do
    echo "Pass $x"
    feats="ark:scripts/subset_scp.pl ${subset[$x]} data/train.scp | add-deltas --print-args=false scp:- ark:- |"
    fgmm-acc-stats --diag-gmm-nbest=15 --binary=false --verbose=2 $dir/$x.ubm "$feats" $dir/$x.acc \
	2> $dir/acc.$x.log  || exit 1;
    fgmm-est --verbose=2 $dir/$x.ubm $dir/$x.acc \
	$dir/$[$x+1].ubm 2> $dir/update.$x.log || exit 1;
    rm $dir/$x.acc $dir/$x.ubm
done



