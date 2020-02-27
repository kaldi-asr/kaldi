#!/usr/bin/env bash

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

# Decoding script that works with a GMM model and delta-delta plus
# cepstral mean subtraction features.  Used, for example, to decode
# mono/ and tri1/
# This script just generates lattices for a single broken-up
# piece of the data.

if [ -f ./path.sh ]; then . ./path.sh; fi

numjobs=1
jobid=0
if [ "$1" == "-j" ]; then
  shift;
  numjobs=$1;
  jobid=$2;
  shift; shift;
fi

if [ $# != 3 ]; then
   echo "Usage: steps/decode_deltas.sh [-j num-jobs job-number] <graph-dir> <data-dir> <decode-dir>"
   echo " e.g.: steps/decode_deltas.sh -j 8 0 exp/mono/graph_tgpr data/test_dev93 exp/mono/decode_dev93_tgpr"
   exit 1;
fi


graphdir=$1
data=$2
dir=$3
srcdir=`dirname $dir`; # Assume model directory one level up from decoding directory.

mkdir -p $dir

if [ $numjobs -gt 1 ]; then
  mydata=$data/split$numjobs/$jobid
else
  mydata=$data
fi

requirements="$mydata/feats.scp $srcdir/final.mdl $graphdir/HCLG.fst"
for f in $requirements; do
  if [ ! -f $f ]; then
     echo "decode_deltas.sh: no such file $f";
     exit 1;
  fi
done


# We only do one decoding pass, so there is no point caching the
# CMVN stats-- we make them part of a pipe.
feats="ark:compute-cmvn-stats --spk2utt=ark:$mydata/spk2utt scp:$mydata/feats.scp ark:- | apply-cmvn --norm-vars=false --utt2spk=ark:$mydata/utt2spk ark:- scp:$mydata/feats.scp ark:- | add-deltas ark:- ark:- |"

gmm-latgen-faster --max-active=7000 --beam=13.0 --lattice-beam=6.0 --acoustic-scale=0.083333 \
  --allow-partial=true --word-symbol-table=$graphdir/words.txt \
  $srcdir/final.mdl $graphdir/HCLG.fst "$feats" "ark:|gzip -c > $dir/lat.$jobid.gz" \
  2>> $dir/decode$jobid.log || exit 1;

