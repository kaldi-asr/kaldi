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
   echo "Usage: scripts/decode_sgmm2b_fmllr.sh <graph> <decode-dir> <job-number>"
   exit 1;
fi

. path.sh || exit 1;

acwt=0.08333
prebeam=12.0
beam=13.0
max_active=7000
silphones=`cat data/silphones.csl`
model=exp/sgmm2b/final.mdl
occs=exp/sgmm2b/final.occs
alimodel=exp/sgmm2b/final.alimdl
fmllr_model=exp/sgmm2b/final_fmllr.mdl
graph=$1
dir=$2
job=$3
scp=$dir/$job.scp
feats="ark:add-deltas --print-args=false scp:$scp ark:- |"

if [ ! -f $fmllr_model ]; then
  if [ ! -f $model ]; then
    echo "Cannot find $model. Maybe training didn't finish?"
    exit 1;
  fi
  sgmm-comp-prexform $model $occs $fmllr_model
fi


filenames="$scp $alimodel $fmllr_model $model $graph data/words.txt"
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
     ark:$dir/${job}.vecs1 ) 2>$dir/vecs1.${job}.log || exit 1;

( ali-to-post ark:$dir/${job}.pre_ali ark:- | \
  weight-silence-post 0.01 $silphones $alimodel ark:- ark:- | \
  sgmm-est-spkvecs --spk-vecs=ark,t:$dir/${job}.vecs1 $spk2utt_opt $model \
   "$feats" ark,s,cs:- ark:$dir/${job}.vecs2 ) 2>$dir/vecs2.${job}.log || exit 1;

# second pass of decoding: have spk-vecs but not fMLLR
sgmm-decode-faster "$gselect_opt" --beam=$prebeam --max-active=$max_active \
   $utt2spk_opt --spk-vecs=ark:$dir/${job}.vecs2 \
   --acoustic-scale=$acwt \
   --word-symbol-table=data/words.txt $model $graph "$feats" \
   ark,t:$dir/$job.pre2_tra ark,t:$dir/$job.pre2_ali  2>$dir/pre2decode${job}.log  || exit 1;


# Estimate fMLLR transforms.

( ali-to-post ark:$dir/$job.pre2_ali ark:- | \
    weight-silence-post 0.01 $silphones $model ark:- ark:- | \
    sgmm-post-to-gpost "$gselect_opt" $model "$feats" ark,s,cs:- ark:- | \
    sgmm-est-fmllr-gpost --spk-vecs=ark:$dir/${job}.vecs2 $spk2utt_opt $fmllr_model "$feats" ark,s,cs:- \
	ark:$dir/$job.fmllr ) 2>$dir/est_fmllr${job}.log

feats="ark:add-deltas --print-args=false scp:$scp ark:- | transform-feats $utt2spk_opt ark:$dir/$job.fmllr ark:- ark:- |"

sgmm-decode-faster "$gselect_opt" $utt2spk_opt --spk-vecs=ark:$dir/${job}.vecs2 \
     --beam=$beam --acoustic-scale=$acwt --word-symbol-table=data/words.txt \
     $fmllr_model $graph "$feats" \
    ark,t:$dir/${job}.tra ark,t:$dir/${job}.ali  2> $dir/decode${job}.log

