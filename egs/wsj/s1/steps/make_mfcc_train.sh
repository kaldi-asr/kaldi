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

# To be run from .. (one directory up from here)

if [ $# != 1 ]; then
    echo "usage: make_mfcc_train.sh <abs-path-to-tmpdir>";
    exit 1;
fi

if [ -f path.sh ]; then . path.sh; fi

scpin=data/train_wav.scp   
dir=exp/make_mfcc
mkdir -p $dir
root_out=$1

mkdir -p $root_out

scripts/split_scp.pl $scpin $dir/train_wav{1,2,3,4}.scp

# Making it like this so it works for others on the BUT filesystem.
# It will generate the correct scp file without running the feature extraction.
for n in 1 2 3 4; do # Use 4 CPUs
   compute-mfcc-feats  --verbose=2 --config=conf/mfcc.conf scp:$dir/train_wav${n}.scp  ark,scp:$root_out/train_raw_mfcc${n}.ark,$root_out/train_raw_mfcc${n}.scp  2> $dir/make_mfcc_train.${n}.log || echo Error &
done
wait;

cat $root_out/train_raw_mfcc{1,2,3,4}.scp > data/train.scp

