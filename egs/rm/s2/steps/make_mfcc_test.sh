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
   echo "usage: make_mfcc_test.sh <abs-path-to-tmpdir>"
   exit 1;
fi

if [ -f path.sh ]; then . path.sh; fi

dir=exp/make_mfcc
mkdir -p $dir
root_out=$1
mkdir -p $root_out

for test in mar87 oct87 feb89 oct89 feb91 sep92; do
  scpin=data_prep/test_${test}_wav.scp 
# Making it like this so it works for others on the BUT filesystem.
# It will generate the correct scp file without running the feature extraction.
  log=$dir/make_mfcc_test_${test}.log
  (
    compute-mfcc-feats  --verbose=2 --config=conf/mfcc.conf scp:$scpin ark,scp:$root_out/test_${test}_raw_mfcc.ark,$root_out/test_${test}_raw_mfcc.scp  2> $log || tail $log
    cp $root_out/test_${test}_raw_mfcc.scp data/test_${test}.scp
  ) &
done

wait

echo "If the above produced no output on the screen, it succeeded."
