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


if [ $# != 3 ]; then
   echo "Usage: scripts/decode_sgmm3b.sh <graph> <decode-dir> <job-number>"
   exit 1;
fi

. path.sh || exit 1;

acwt=0.08333
prebeam=12.0
beam=13.0
max_active=7000
silphones=`cat data/silphones.csl`
model=exp/sgmm3b/final.mdl
alimodel=exp/sgmm3b/final.alimdl
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

if [ -f $dir/$job.spk2utt ]; then
  if [ ! -f $dir/$job.utt2spk ]; then
     echo "spk2utt but not utt2spk file present!"
     exit 1
  fi
  spk2utt_opt=--spk2utt=ark:$dir/$job.spk2utt
  utt2spk_opt=--utt2spk=ark:$dir/$job.utt2spk
fi

echo running on `hostname` > $dir/decode${job}.log


sgmm-gselect $model "$feats" ark,t:- 2>$dir/gselect${job}.log | \
     gzip -c > $dir/gselect${job}.gz || exit 1;
gselect_opt="--gselect=ark:gunzip -c $dir/gselect${job}.gz|"


sgmm-decode-faster "$gselect_opt" --beam=$prebeam --max-active=$max_active \
   --acoustic-scale=$acwt \
   --word-symbol-table=data/words.txt $alimodel $graph "$feats" \
   ark,t:$dir/$job.pre_tra ark,t:$dir/$job.pre_ali  2>$dir/predecode${job}.log  || exit 1;

( ali-to-post ark:$dir/${job}.pre_ali ark:- | \
  weight-silence-post 0.01 $silphones $alimodel ark:- ark:- | \
  sgmm-post-to-gpost "$gselect_opt" $alimodel "$feats" ark,s,cs:- ark:- | \
  sgmm-est-spkvecs-gpost $spk2utt_opt $model "$feats" ark,s,cs:- \
     ark:$dir/${job}.vecs ) 2>$dir/vecs${job}.log || exit 1;

sgmm-decode-faster "$gselect_opt" --beam=$beam --max-active=$max_active \
   $utt2spk_opt --spk-vecs=ark:$dir/${job}.vecs \
   --acoustic-scale=$acwt \
   --word-symbol-table=data/words.txt $model $graph "$feats" \
   ark,t:$dir/$job.tra ark,t:$dir/$job.ali  2>$dir/decode${job}.log  || exit 1;


