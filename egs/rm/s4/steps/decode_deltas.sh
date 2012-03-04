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
# This script generates lattices and rescores them with different
# acoustic weights, in order to explore a range of different
# weights.

if [ $# != 4 ]; then
   echo "Usage: steps/decode_deltas.sh <model-dir> <data-dir> <lang-dir> <decode-dir>"
   echo " e.g.: steps/decode_deltas.sh exp/mono data/test_feb89 data/lang_test exp/mono/decode/feb89"
   exit 1;
fi

srcdir=$1
data=$2
lang=$3
dir=$4
graphdir=$srcdir/graph

mkdir -p $dir

if [ -f path.sh ]; then . path.sh; fi

if [ ! -f $srcdir/final.mdl ]; then
   echo No model file $srcdir/final.mdl
   exit 1;
fi

if [[ ! -f $graphdir/HCLG.fst || $graphdir/HCLG.fst -ot $srcdir/final.mdl ]]; then
   echo "Graph $graphdir/HCLG.fst does not exist or is too old."
   exit 1;
fi

# We only do one decoding pass, so there is no point caching the
# CMVN stats-- we make them part of a pipe.
feats="ark:compute-cmvn-stats scp:$data/feats.scp ark:- | apply-cmvn --norm-vars=false  ark:- scp:$data/feats.scp ark:- | add-deltas ark:- ark:- |"

# For Resource Management, we use beam of 20 and acwt of 1/10.
# More normal, LVCSR setups would have a beam of 13 and acwt of 1/15 or so.
# If you decode with a beam of 20 on an LVCSR setup it will be very slow.

gmm-latgen-simple --beam=20.0 --acoustic-scale=0.1 --word-symbol-table=$lang/words.txt \
  $srcdir/final.mdl $graphdir/HCLG.fst "$feats" "ark:|gzip -c > $dir/lat.gz" \
  ark,t:$dir/test.tra ark,t:$dir/test.ali \
     2> $dir/decode.log || exit 1;

# In this setup there are no non-scored words, so
# scoring is simple.

# Now rescore lattices with various acoustic scales, and compute the WER.
for inv_acwt in 4 5 6 7 8 9 10; do
  acwt=`perl -e "print (1.0/$inv_acwt);"`
  lattice-best-path --acoustic-scale=$acwt --word-symbol-table=$lang/words.txt \
     "ark:gunzip -c $dir/lat.gz|" ark,t:$dir/${inv_acwt}.tra \
     2>$dir/rescore_${inv_acwt}.log

  scripts/sym2int.pl --ignore-first-field $lang/words.txt $data/text | \
   compute-wer --mode=present ark:-  ark,p:$dir/${inv_acwt}.tra \
    >& $dir/wer_${inv_acwt}
done
