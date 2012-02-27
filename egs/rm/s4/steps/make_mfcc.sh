#!/bin/bash
# Copyright 2012 Vassil Panayotov <vd.panayotov@gmail.com>
#
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

echo "--- Start preparing MFCC for $2 ..."

if [ $# != 2 ]; then
    echo "usage: make_mfcc_train.sh <destination_dir> <train OR test>";
    exit 1;
fi

if [ -f path.sh ]; then . path.sh; fi

scpin=data_prep/$2.scp  
out=$1
mkdir -p $out

copy-feats --sphinx-in=true scp:$scpin ark,scp:$out/$2.ark,$out/$2.scp
cp $out/$2.scp data/$2.scp

echo "--- Done preparing MFCC for $2"
