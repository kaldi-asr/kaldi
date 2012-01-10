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

for test in eval_nov92 dev_nov93 eval_nov93; do
  rm data/${test}.scp 2>/dev/null
  scpin=data_prep/${test}_wav.scp
  dir=exp/make_mfcc
  mkdir -p $dir
  root_out=$1
  mkdir -p $root_out
  ( compute-mfcc-feats  --verbose=2 --config=conf/mfcc.conf scp:$scpin ark,scp:$root_out/${test}_raw_mfcc.ark,$root_out/${test}_raw_mfcc.scp  2> $dir/make_mfcc_${test}.log || exit 1;
   cp $root_out/${test}_raw_mfcc.scp data/${test}.scp ) &
done
wait

for test in eval_nov92 dev_nov93 eval_nov93; do
  if [ ! -f data/${test}.scp ];  then
     echo Failed to make test MFCC features for at least test-set $test
   fi
done

echo Succeeded "(probably)"

