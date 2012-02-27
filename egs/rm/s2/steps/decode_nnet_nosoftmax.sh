#!/bin/bash

# Copyright 2012 Karel Vesely

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

# Decoding script using pure-hybrid acoustic model, 
# using fbank features with cepstral mean normalization 
# and hamming-dct transform

if [ "$1" == "--acoustic-scale" ]; then
  shift;
  acousticscale=$1
  shift;
fi

if [ "$1" == "--prior-scale" ]; then
  shift;
  priorscale=$1
  shift;
fi



if [ $# != 4 ]; then
   echo "Usage: steps/decode_deltas.sh <model-dir> <data-dir> <lang-dir> <decode-dir>"
   echo " e.g.: steps/decode_deltas.sh exp/mono data/test_feb89 data/test_lang exp/mono/decode_feb89"
   exit 1;
fi

srcdir=$1
data=$2
lang=$3
dir=$4
graphdir=$srcdir/graph

######## CONFIGURATION
beam=30
norm_vars=false #false:CMN, true:CMVN on fbanks
splice_lr=15   #left- and right-splice value
transf=$srcdir/hamm_dct.mat
cmvn_g=$srcdir/cmvn_glob.mat
priors=$srcdir/cur.counts
nnet=$srcdir/final.nnet
########

mkdir -p $dir

if [ -f path.sh ]; then . path.sh; fi

if [ ! -f $srcdir/final.nnet ]; then
   echo No model file $srcdir/final.nnet
   exit 1;
fi

if [[ ! -f $graphdir/HCLG.fst || $graphdir/HCLG.fst -ot $srcdir/transition.mdl ]]; then
   echo "Graph $graphdir/HCLG.fst does not exist or is too old."
   exit 1;
fi

#get rid of softmax
grep -v "<softmax>" $nnet > $dir/${nnet##*/}.no-softmax
nnet=$dir/${nnet##*/}.no-softmax

# prepare features
# We only do one decoding pass, so there is no point caching the
# CMVN stats-- we make them part of a pipe.
feats="ark:compute-cmvn-stats --spk2utt=ark:$data/spk2utt scp:$data/feats.scp.fbank ark:- | apply-cmvn --norm-vars=$norm_vars --utt2spk=ark:$data/utt2spk ark:- scp:$data/feats.scp.fbank ark:- |"
# Splice+Hamming+DCT
feats="$feats splice-feats --print-args=false --left-context=$splice_lr --right-context=$splice_lr ark:- ark:- | transform-feats --print-args=false $transf ark:- ark:- |"
# Norm+MLP
feats="$feats apply-cmvn --print-args=false --norm-vars=true $cmvn_g ark:- ark:- | nnet-forward --silent=false --print-args=false --no-softmax=true --class-frame-counts=$priors ${priorscale:+--prior-scale=$priorscale} $nnet ark:- ark:- |"

# For Resource Management, we use beam of 30 and acwt of 1/7.
# More normal, LVCSR setups would have a beam of 13 and acwt of 1/15 or so.
# If you decode with a beam of 20 on an LVCSR setup it will be very slow.
decode-faster-mapped --beam=$beam --acoustic-scale=${acousticscale:-0.1429} --word-symbol-table=$lang/words.txt \
  $srcdir/transition.mdl $graphdir/HCLG.fst "$feats" ark,t:$dir/test.tra ark,t:$dir/test.ali \
     2> $dir/decode.log || exit 1;




# In this setup there are no non-scored words, so
# scoring is simple.

# the ,p option lets it score partial output without dying..
scripts/sym2int.pl --ignore-first-field $lang/words.txt $data/text | \
  compute-wer --mode=present ark:-  ark,p:$dir/test.tra >& $dir/wer




