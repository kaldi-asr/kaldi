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

# Note: this file is not used in this set of scripts.
# It's intended to demonstrate to Martin how to work with segmentations.
# It will eventually be deleted.
# To be run from .. (one directory up from here)


if [ -f path.sh ]; then . path.sh; fi

scpin=data_prep/train_wav.scp  
dir=exp/make_mfcc
mkdir -p $dir
root_out=$dir


head -10 $scpin > $dir/tmp_wav.scp

# now make segments file.
cat $dir/tmp_wav.scp | awk '{ printf("%s_seg1 %s 0.5 1.34\n", $1, $1); printf("%s_seg2 %s 1.45 2.06\n", $1, $1); }' > $dir/segments

extract-segments scp:$dir/tmp_wav.scp $dir/segments ark:- | \
   compute-mfcc-feats  --verbose=2 --config=conf/mfcc.conf ark:- \
 ark,scp:$root_out/train_raw_mfcc.ark,$root_out/train_raw_mfcc.scp 


# now make segments file, with channel zero specified
cat $dir/tmp_wav.scp | awk '{ printf("%s_seg1 %s 0.5 1.34 0\n", $1, $1); printf("%s_seg2 %s 1.45 2.06 0\n", $1, $1); }' > $dir/segments

extract-segments scp:$dir/tmp_wav.scp $dir/segments ark:- | \
   compute-mfcc-feats  --verbose=2 --config=conf/mfcc.conf ark:- \
 ark,scp:$root_out/train_raw_mfcc.ark,$root_out/train_raw_mfcc.scp 


# The following example shows how we would use "extract-segments" to extract
# the segments as wave files and write them to disk [probably not useful,
# but shows the ideas.
mkdir ~/tmpdir
cat $dir/segments | awk '{printf("%s /homes/eva/q/qpovey/tmpdir/%s.wav\n", $1, $1);}' | extract-segments scp:$dir/tmp_wav.scp $dir/segments scp:- 

