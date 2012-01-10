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

acwt=0.0625
#acwt=0.05
#acwt=0.04
#acwt=0.03
#acwt=0.04

beam=13.0
#beam=20.0
max_active=7000
#max_active=14000

model=exp/nnet-bn_tri3a_s4b_tri2j.gmm/final.mdl
mat=exp/nnet-bn_tri3a_s4b_tri2j.gmm/final.mat
graph=$1
dir=$2
job=$3
scp=$dir/$job.scp

######### 
nndir=exp/nnet-bn_tri3a_s4b_net/
cvn=$nndir/global_cvn.mat
nnet=exp/nnet-bn_tri3a_s4b_tri2j.gmm/nnet_fwd.nnet

######### Compose features-to-bnfeatures pipeline
feats="ark:splice-feats --print-args=false --left-context=15 --right-context=15 scp:$scp ark:- |"
feats="$feats transform-feats $nndir/lda.mat ark:- ark:- |"
######### compute cmn
cmn=ark:$dir/job${job}_cmn.ark
compute-cmvn-stats "$feats" $cmn
feats="$feats apply-cmvn --print-args=false --norm-vars=false $cmn ark:- ark:- |"
######### add cvn
feats="$feats apply-cmvn --print-args=false --norm-vars=true $cvn ark:- ark:- |"
######### add nnet
feats="$feats nnet-forward --print-args=false $nnet ark:- ark:- |"
######### add deltas
feats="$feats add-deltas --print-args=false --delta-order=1 ark:- ark:- |" 
######### add MLLT
feats="$feats transform-feats $mat ark:- ark:- |"

filenames="$scp $model $graph data/words.txt"
for file in $filenames; do
  if [ ! -f $file ] ; then
    echo "No such file $file" >&2;
    echo "No such file $file"; 
    exit 1;
  fi
done

echo running on `hostname` > $dir/decode${job}.log
gmm-decode-faster --beam=$beam --max-active=$max_active --acoustic-scale=$acwt --word-symbol-table=data/words.txt $model $graph "$feats" ark,t:$dir/$job.tra ark,t:$dir/$job.ali  2>>$dir/decode${job}.log 

