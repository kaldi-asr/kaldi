#!/bin/bash
# Copyright 2012   Carnegie Mellon University  Yajie Miao

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


# This script does the decoding similar with decode_tri2a_fmllr.sh, but with
# basis-fMLLR, rather than the traditional fMLLR. We need to specify pointer
# to fMLLR base matrices, which have been estimated with train_tri2a_fmllr_basis.sh.
# Refer to decode_tri2a_fmllr.sh for details on other arguments.
# 

if [ $# != 3 ]; then
   echo "Usage: steps/decode_tri2a_fmllr.sh <graph> <decode-dir> <job-number>"
   exit 1;
fi

. path.sh || exit 1;

acwt=0.0625
beam=13.0
prebeam=12.0 # first-pass decoding beam...
max_active=7000
# base matrices trained with train_tri2a_fmllr_basis.sh prior to decoding
fmllrbasis=exp/tri2a_fmllr_basis/fmllr.base.mats
model=exp/tri2a/final.mdl
silphones=`cat data/silphones.csl`
mincount=500 # mincount before we estimate a transform.
graph=$1
dir=$2
job=$3
scp=$dir/$job.scp
sifeats="ark:add-deltas --print-args=false scp:$scp ark:- |"
if [ -f $dir/$job.spk2utt ]; then
  if [ ! -f $dir/$job.utt2spk ]; then
     echo "spk2utt but not utt2spk file present!"
     exit 1
  fi
  spk2utt_opt=--spk2utt=ark:$dir/$job.spk2utt
  utt2spk_opt=--utt2spk=ark:$dir/$job.utt2spk
fi

filenames="$scp $model $graph data/words.txt"
for file in $filenames; do
  if [ ! -f $file ] ; then
    echo "No such file $file";
    exit 1;
  fi
done

echo running on `hostname` > $dir/predecode${job}.log

# First-pass decoding.
gmm-decode-faster --beam=$prebeam --max-active=$max_active --acoustic-scale=$acwt --word-symbol-table=data/words.txt $model $graph "$sifeats" ark,t:$dir/$job.pre_tra ark,t:$dir/$job.pre_ali  2>>$dir/predecode${job}.log 

# Estimate basis fMLLR transforms
ali-to-post ark:$dir/$job.pre_ali ark:- | \
  weight-silence-post 0.0 $silphones $model ark:- ark:- | \
  gmm-est-basis-fmllr $spk2utt_opt --fmllr-min-count=$mincount --num-iters=10 \
  --size-scale=0.2 --step-size-iters=3 --write-weights=ark:$dir/$job.wgt \
  $model $fmllrbasis "$sifeats" ark,o:- ark:$dir/$job.fmllr 2>$dir/fmllr${job}.log

feats="ark:add-deltas --print-args=false scp:$scp ark:- | transform-feats $utt2spk_opt ark:$dir/$job.fmllr ark:- ark:- |"

# Final decoding with adaptation
echo running on `hostname` > $dir/decode${job}.log
gmm-decode-faster --beam=$beam --max-active=$max_active --acoustic-scale=$acwt --word-symbol-table=data/words.txt $model $graph "$feats" ark,t:$dir/$job.tra ark,t:$dir/$job.ali  2>>$dir/decode${job}.log 
