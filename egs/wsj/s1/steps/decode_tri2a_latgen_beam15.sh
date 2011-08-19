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


# This script does the decoding of a single batch of test data (on one core).
# It requires arguments.  It takes the graphdir and decoding directory,
# and the job number which can actually be any string (even ""); it expects
# a file $decode_dir/test${job_number}.scp to exist, and puts its output in
# $decode_dir/${job_number}.tra

# This script also creates the text archive $decode_dir/${job_number}.lats.gz

if [ $# != 3 ]; then
   echo "Usage: steps/decode_tri2a_latgen.sh <graph> <decode-dir> <job-number>"
   exit 1;
fi

. path.sh || exit 1;

acwt=0.0625
beam=15.0
max_active=15000
model=exp/tri2a/final.mdl
graph=$1
dir=$2
job=$3
scp=$dir/$job.scp
feats="ark:add-deltas --print-args=false scp:$scp ark:- |"

filenames="$scp $model $graph data/words.txt"
for file in $filenames; do
  if [ ! -f $file ] ; then
    echo "No such file $file";
    exit 1;
  fi
done

echo running on `hostname` > $dir/decode${job}.log
gmm-latgen-simple --beam=$beam --acoustic-scale=$acwt --word-symbol-table=data/words.txt $model $graph "$feats" "ark,t:|gzip -c >$dir/$job.lats.gz" ark,t:$dir/$job.tra ark,t:$dir/$job.ali  2>>$dir/decode${job}.log 

