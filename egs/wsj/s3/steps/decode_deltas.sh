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

# Decoding script that works with a GMM model and delta-delta plus
# cepstral mean subtraction features.  Used, for example, to decode
# mono/ and tri1/
# This script just generates lattices for a single broken-up
# piece of the data.

#nd rescores them with different
# acoustic weights, in order to explore a range of different
# weights.

if [ -f ./path.sh ]; then . ./path.sh; fi

numjobs=1
jobid=0
if [ "$1" == "-j" ]; then
  shift;
  numjobs=$1;
  jobid=$2;
  shift; shift;
  if [ $jobid -ge $numjobs ]; then
     echo "Invalid job number, $jobid >= $numjobs";
     exit 1;
  fi
fi

if [ $# != 3 ]; then
   echo "Usage: steps/decode_deltas.sh [-j num-jobs job-number] <graph-dir> <data-dir> <decode-dir>"
   echo " e.g.: steps/decode_deltas.sh -j 8 0 exp/mono/graph_tgpr data/dev_nov93 exp/mono/decode_dev93_tgpr"
   exit 1;
fi


graphdir=$1
data=$2
dir=$3
srcdir=`dirname $dir`; # Assume model directory one level up from decoding directory.

mkdir -p $dir

requirements="$data/feats.scp $srcdir/final.mdl $graphdir/HCLG.fst"
for f in $requirements; do
  if [ ! -f $f ]; then
     echo "decode_deltas.sh: no such file $f";
     exit 1;
  fi
done

# Make the split .scp files...

scripts/split_scp.pl -j $numjobs $jobid --utt2spk=$data/utt2spk $data/feats.scp > $dir/$jobid.scp
scripts/split_scp.pl -j $numjobs $jobid --utt2spk=$data/utt2spk $data/utt2spk > $dir/$jobid.utt2spk
scripts/utt2spk_to_spk2utt.pl $dir/$jobid.utt2spk > $dir/$jobid.spk2utt



# We only do one decoding pass, so there is no point caching the
# CMVN stats-- we make them part of a pipe.
feats="ark:compute-cmvn-stats --spk2utt=ark:$dir/$jobid.spk2utt scp:$dir/$jobid.scp ark:- | apply-cmvn --norm-vars=false --utt2spk=ark:$dir/$jobid.utt2spk ark:- scp:$dir/$jobid.scp ark:- | add-deltas ark:- ark:- |"


gmm-latgen-faster --max-active=7000 --beam=13.0 --lattice-beam=6.0 --acoustic-scale=0.083333 \
  --allow-partial=true --word-symbol-table=$graphdir/words.txt \
  $srcdir/final.mdl $graphdir/HCLG.fst "$feats" "ark:|gzip -c > $dir/lat.$jobid.gz" \
  ark,t:$dir/test.tra ark,t:$dir/test.ali \
     2> $dir/decode$jobid.log || exit 1;


# # In this setup there are no non-scored words, so
# # scoring is simple.

# # Now rescore lattices with various acoustic scales, and compute the WER.
# for inv_acwt in 4 5 6 7 8 9 10; do
#   acwt=`perl -e "print (1.0/$inv_acwt);"`
#   lattice-best-path --acoustic-scale=$acwt --word-symbol-table=$graphdir/words.txt \
#      "ark:gunzip -c $dir/lat.gz|" ark,t:$dir/${inv_acwt}.tra \
#      2>$dir/rescore_${inv_acwt}.log

#   scripts/sym2int.pl --ignore-first-field $graphdir/words.txt $data/text | \
#    compute-wer --mode=present ark:-  ark,p:$dir/${inv_acwt}.tra \
#     >& $dir/wer_${inv_acwt}
# done
