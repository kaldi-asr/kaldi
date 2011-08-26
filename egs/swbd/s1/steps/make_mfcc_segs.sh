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

if [ $# != 4 ]; then
   echo "usage: make_mfcc.sh <data-dir> <log-dir> <abs-path-to-mfccdir> <num-cpus>";
   exit 1;
fi

if [ -f path.sh ]; then . path.sh; fi

data=$1
logdir=$2
mfccdir=$3
ncpus=$4

# use "name" as part of name of the archive.
name=`basename $data`

mkdir -p $mfccdir || exit 1;
mkdir -p $logdir || exit 1;

scp=$data/wav.scp
segments=$data/segments_all  
config=conf/mfcc.conf
required="$scp $segments $config"

for f in $required; do
  if [ ! -f $f ]; then
    echo "make_mfcc.sh: no such file $f"
    exit 1;
  fi
done

# note: in general, the double-parenthesis construct in bash "((" is "C-style
# syntax" where we can get rid of the $ for variable names, and omit spaces.
# The "for" loop in this style is a special construct.

split_segments=""
for ((n=1; n<=ncpus; n++)); do
   split_segments="$split_segments $logdir/segments$n"
done

scripts/split_scp.pl $segments $split_segments || exit 1;

rm $logdir/.error 2>/dev/null
for ((n=1; n<=ncpus; n++)); do
  log=$logdir/make_mfcc.$n.log

  extract-segments scp:$scp $logdir/segments${n} ark:- | \
  compute-mfcc-feats  --sample-frequency=8000 --verbose=2 --config=$config ark:- \
  ark,scp:$mfccdir/raw_mfcc_$name.$n.ark,$mfccdir/raw_mfcc_$name.$n.scp \
  2> $log || touch $logdir/.error &
done
wait;

if [ -f $logdir/.error.$name ]; then
  echo "Error producing mfcc features for $name:"
  tail $logdir/make_mfcc.*.log
  exit 1;
fi

# concatenate the .scp files together.
rm $data/feats.scp 2>/dev/null
for ((n=1; n<=ncpus; n++)); do
  cat $mfccdir/raw_mfcc_$name.$n.scp >> $data/feats.scp
done

rm $logdir/segments*

echo "Succeeded creating MFCC features for $name"

