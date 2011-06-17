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

dir=exp/ubm3a
mkdir -p $dir
srcdir=exp/tri1

init-ubm --intermediate-numcomps=2000 --ubm-numcomps=600 --verbose=2 \
    --fullcov-ubm=true $srcdir/final.mdl $srcdir/final.occs \
    $dir/0.ubm 2> $dir/cluster.log

# Use same subset as for training tri1.
head -7138 data/train.scp | scripts/subset_scp.pl 3500  - > $dir/train.scp
scripts/split_scp.pl $dir/train.scp  $dir/train{1,2,3}.scp

subset[0]=2000
subset[1]=3000
subset[2]=4000
subset[3]=5000

rm -f $dir/.error

for x in 0 1 2 3; do
    echo "Pass $x"
    for n in 1 2 3; do
      feats="ark:add-deltas scp:$dir/train${n}.scp ark:- |"
      fgmm-global-acc-stats --diag-gmm-nbest=15 --binary=false --verbose=2 $dir/$x.ubm "$feats" \
        $dir/$x.$n.acc 2> $dir/acc.$x.$n.log  || touch $dir/.error &
    done
    wait;
    [ -f $dir/.error ] && echo "Error accumulating stats" && exit 1;
    ( fgmm-global-sum-accs - $dir/$x.{1,2,3}.acc | \
     fgmm-global-est --verbose=2 $dir/$x.ubm - $dir/$[$x+1].ubm ) 2> $dir/update.$x.log || exit 1;
    rm $dir/$x.{1,2,3}.acc $dir/$x.ubm
done

rm $dir/final.ubm 2>/dev/null
ln -s 4.ubm $dir/final.ubm

