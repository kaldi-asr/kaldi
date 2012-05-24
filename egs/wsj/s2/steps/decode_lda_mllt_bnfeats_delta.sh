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

# Decoding script that works with a GMM model and the baseline
# [e.g. MFCC] features plus cepstral mean subtraction plus
# LDA+MLLT or similar transform.
# This script just generates lattices for a single broken-up
# piece of the data.

if [ -f ./path.sh ]; then . ./path.sh; fi

numjobs=1
jobid=0
while [ 1 ]; do
  case $1 in
    -j)
      shift; numjobs=$1; jobid=$2;
      shift; shift;
      ;;
    --scale-beams)
      shift; scalebeams=$1; shift;
      ;;
    --beam)
      shift; beam=$1; shift; 
      ;;
    --lattice-beam)
      shift; latticebeam=$1; shift;
      ;;
    --acoustic-scale)
      shift; acousticscale=$1; shift;
      ;;
    *)
      break;
      ;;
  esac
done

if [ $# != 3 ]; then
   echo "Usage: steps/decode_lda_mllt.sh [-j num-jobs job-number] <graph-dir> <data-dir> <decode-dir>"
   echo " e.g.: steps/decode_lda_mllt.sh -j 8 0 exp/tri2b/graph_tgpr data/test_dev93 exp/tri2b/decode_dev93_tgpr"
   exit 1;
fi


graphdir=$1
data=$2
dir=$3
srcdir=`dirname $dir`; # Assume model directory one level up from decoding directory.

#default values
echo scalebeams: ${scalebeams:=1.0}
echo beam: ${beam:=$(awk "BEGIN{ print 13.0*$scalebeams; }")}
echo latticebeam: ${latticebeam:=$(awk "BEGIN{ print 6.0*$scalebeams; }")}
echo acousticscale: ${acousticscale:=0.083333}


mkdir -p $dir

if [ $numjobs -gt 1 ]; then
  mydata=$data/split$numjobs/$jobid
else
  mydata=$data
fi

requirements="$mydata/feats.scp $srcdir/final.mdl $srcdir/final.mat $graphdir/HCLG.fst"
for f in $requirements; do
  if [ ! -f $f ]; then
     echo "decode_lda_mllt.sh: no such file $f";
     exit 1;
  fi
done


# We only do one decoding pass, so there is no point caching the
# CMVN stats-- we make them part of a pipe.
feats="ark:compute-cmvn-stats --spk2utt=ark:$mydata/spk2utt scp:$mydata/feats.scp ark:- | apply-cmvn --norm-vars=false --utt2spk=ark:$mydata/utt2spk ark:- scp:$mydata/feats.scp ark:- | add-deltas --delta-order=1 ark:- ark:- | transform-feats $srcdir/final.mat ark:- ark:- |"

gmm-latgen-faster --max-active=7000 --beam=${beam} --lattice-beam=${latticebeam} --acoustic-scale=${acousticscale} \
  --allow-partial=true --word-symbol-table=$graphdir/words.txt \
  $srcdir/final.mdl $graphdir/HCLG.fst "$feats" "ark:|gzip -c > $dir/lat.$jobid.gz" \
     2> $dir/decode$jobid.log || exit 1;

