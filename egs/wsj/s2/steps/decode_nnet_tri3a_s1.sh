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
   echo "Usage: scripts/decode_tri3a.sh <graph> <decode-dir> <job-number>"
   exit 1;
fi

. path.sh || exit 1;

acwt=0.155
beam=13.0
max_active=7000
model=exp/reduce_pdf_count/final.mdl

dir_nnet=exp/nnet_tri3a_s1
nnet=$dir_nnet/final.nnet
priors=$dir_nnet/cur.counts #optional
cvn=$dir_nnet/global_cvn.mat

graph=$1
dir=$2
job=$3
scp=$dir/$job.scp


filenames="$scp $model $nnet $priors $cvn $graph data/words.txt"
for file in $filenames; do
  if [ ! -f $file ] ; then
    echo "No such file $file";
    exit 1;
  fi
done

######### Compose features-to-posteriors pipeline
feats="ark:add-deltas --print-args=false scp:$scp ark:- |"

#compute per-utterance CMN
cmn=ark:$dir/job${job}_cmn.ark
compute-cmvn-stats "$feats" $cmn
feats="$feats apply-cmvn --print-args=false --norm-vars=false $cmn ark:- ark:- |"

#add precomputed global CVN 
feats="$feats apply-cmvn --print-args=false --norm-vars=true $cvn ark:- ark:- |"

#add splicing
feats="$feats splice-feats --print-args=false --left-context=5 --right-context=5 ark:- ark:- |"

#add MLP transform
feats="$feats nnet-forward --print-args=false --apply-log=true ${priors:+--class-frame-counts=$priors} $nnet ark:- ark:- |"

#########


echo running on `hostname` > $dir/decode${job}.log
decode-faster-mapped --beam=$beam --max-active=$max_active --acoustic-scale=$acwt --word-symbol-table=data/words.txt $model $graph "$feats" ark,t:$dir/$job.tra ark,t:$dir/$job.ali  2>>$dir/decode${job}.log 

