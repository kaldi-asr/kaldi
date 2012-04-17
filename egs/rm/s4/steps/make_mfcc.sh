#!/bin/bash 
# Copyright 2012 Vassil Panayotov

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

if [ $# != 3 ]; then
   echo "usage: make_mfcc.sh <data-dir> <log-dir> <abs-path-to-mfccdir>";
   exit 1;
fi

if [ -f path.sh ]; then . path.sh; fi

data=$1
logdir=$2
mfccdir=$3

# use "name" as part of name of the archive.
name=`basename $data`

mkdir -p $mfccdir || exit 1;
mkdir -p $logdir || exit 1;

scp=$data/mfc.scp
if [ ! -f $scp ]; then
   echo "make_mfcc.sh: no such file $f";
   exit 1;
fi

log=$logdir/make_mfcc.log

copy-feats --sphinx-in=true \
 scp:$scp ark,scp:$mfccdir/raw_mfcc_$name.ark,$data/feats.scp 2>$log

echo "Succeeded creating MFCC features for $name"

