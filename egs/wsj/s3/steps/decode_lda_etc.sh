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

# Decoding script for LDA + optionally MLLT + [some speaker-specific transforms]
# This decoding script takes as an argument a previous decoding directory where it
# can find some transforms.

if [ -f ./path.sh ]; then . ./path.sh; fi

numjobs=1
jobid=0
if [ "$1" == "-j" ]; then
  shift;
  numjobs=$1;
  jobid=$2;
  shift; shift;
fi

if [ $# != 4 ]; then
   # Note: transform-dir has to be last because scripts/decode.sh expects decode-dir to be #3 arg.
   echo "Usage: steps/decode_lda_etc.sh [-j num-jobs job-number] <graph-dir> <data-dir> <decode-dir> <transform-dir>"
   echo " e.g.: steps/decode_lda_etc.sh -j 8 0 exp/tri4b/graph_tgpr data/test_dev93 exp/tri4b_mmi/decode_tgpr_dev93 exp/tri4b/decode_tgpr_dev93"
   exit 1;
fi


graphdir=$1
data=$2
dir=$3
transdir=$4
srcdir=`dirname $dir`; # Assume model directory one level up from decoding directory.

mkdir -p $dir

if [ $numjobs -gt 1 ]; then
  mydata=$data/split$numjobs/$jobid
else
  mydata=$data
fi

requirements="$mydata/feats.scp $srcdir/final.mdl $srcdir/final.mat $graphdir/HCLG.fst $transdir/$jobid.trans"
for f in $requirements; do
  if [ ! -f $f ]; then
     echo "decode_lda_mllt.sh: no such file $f";
     exit 1;
  fi
done


# We only do one decoding pass, so there is no point caching the
# CMVN stats-- we make them part of a pipe.
feats="ark:compute-cmvn-stats --spk2utt=ark:$mydata/spk2utt scp:$mydata/feats.scp ark:- | apply-cmvn --norm-vars=false --utt2spk=ark:$mydata/utt2spk ark:- scp:$mydata/feats.scp ark:- | splice-feats ark:- ark:- | transform-feats $srcdir/final.mat ark:- ark:- | transform-feats --utt2spk=ark:$mydata/utt2spk ark:$transdir/$jobid.trans ark:- ark:- |"

gmm-latgen-faster --max-active=7000 --beam=13.0 --lattice-beam=6.0 --acoustic-scale=0.083333 \
  --allow-partial=true --word-symbol-table=$graphdir/words.txt \
  $srcdir/final.mdl $graphdir/HCLG.fst "$feats" "ark:|gzip -c > $dir/lat.$jobid.gz" \
     2> $dir/decode$jobid.log || exit 1;

