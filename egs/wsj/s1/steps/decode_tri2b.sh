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
# It requires arguments.  It takes the graphdir and decoding directory, and the
# job number.  It expects a file $decode_dir/test${job_number}.scp to exist, and
# puts its output in $decode_dir/${job_number}.tra 
#
# If the files
# $decode_dir/${job_number}.utt2spk and $decode_dir/${job_number}.spk2utt exist,
# this script will assume you want to do per-speaker (not per-utterance) adaptation.

if [ $# != 3 ]; then
   echo "Usage: scripts/decode_tri2b.sh <graph> <decode-dir> <job-number>"
   exit 1;
fi

. path.sh || exit 1;

acwt=0.0625
beam=13.0
prebeam=12.0 # first-pass decoding beam...
max_active=7000
alimodel=exp/tri2b/final.alimdl # first-pass model...
model=exp/tri2b/final.mdl
et=exp/tri2b/final.et
defaultmat=exp/tri2b/default.mat
silphones=`cat data/silphones.csl`
graph=$1
dir=$2
job=$3
scp=$dir/$job.scp
sifeats="ark:add-deltas --print-args=false scp:$scp ark:- |"
defaultfeats="ark:add-deltas --print-args=false scp:$scp ark:- | transform-feats $defaultmat ark:- ark:- |"
if [ -f $dir/$job.spk2utt ]; then
  if [ ! -f $dir/$job.utt2spk ]; then
     echo "spk2utt but not utt2spk file present!"
     exit 1
  fi
  spk2utt_opt=--spk2utt=ark:$dir/$job.spk2utt
  utt2spk_opt=--utt2spk=ark:$dir/$job.utt2spk
fi

filenames="$scp $model $alimodel $et $graph data/words.txt"
for file in $filenames; do
  if [ ! -f $file ] ; then
    echo "No such file $file";
    exit 1;
  fi
done

echo running on `hostname` > $dir/predecode${job}.log

# First-pass decoding 

gmm-decode-faster --beam=$prebeam --max-active=$max_active --acoustic-scale=$acwt --word-symbol-table=data/words.txt $alimodel $graph "$defaultfeats" ark,t:$dir/$job.pre_tra ark,t:$dir/$job.pre_ali  2>>$dir/predecode${job}.log 

# Estimate transforms
( ali-to-post ark:$dir/$job.pre_ali ark:- | \
  weight-silence-post 0.0 $silphones $alimodel ark:- ark:- | \
  gmm-post-to-gpost $alimodel "$defaultfeats" ark,o:- ark:- | \
  gmm-est-et $spk2utt_opt $model $et "$sifeats" ark,o:- \
     ark:$dir/$job.trans ark,t:$dir/$job.warp ) 2>$dir/et${job}.log

feats="ark:add-deltas --print-args=false scp:$scp ark:- | transform-feats $utt2spk_opt ark:$dir/$job.trans ark:- ark:- |"

# Final decoding
echo running on `hostname` > $dir/decode$job.log
gmm-decode-faster --beam=$beam --max-active=$max_active --acoustic-scale=$acwt --word-symbol-table=data/words.txt $model $graph "$feats" ark,t:$dir/$job.tra ark,t:$dir/$job.ali  2>>$dir/decode$job.log 

