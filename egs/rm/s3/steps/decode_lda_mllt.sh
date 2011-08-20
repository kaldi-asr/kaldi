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

# Decoding script that works with a GMM model and cepstral
# mean subtraction plus splice-9-frames plus LDA + mllt features.
# Used, for example, to decode tri2b/.

if [ $# != 4 ]; then
   echo "Usage: steps/decode_ldamllt.sh <model-dir> <data-dir> <lang-dir> <decode-dir>"
   echo " e.g.: steps/decode_ldamllt.sh exp/tri2b data/test_feb89 data/lang_test exp/tri2b/decode_feb89"
   exit 1;
fi

srcdir=$1
data=$2
lang=$3
dir=$4
graphdir=$srcdir/graph

mkdir -p $dir

if [ -f path.sh ]; then . path.sh; fi

# -f means file exists; -o means or.
if [ ! -f $srcdir/final.mdl -o ! -f $srcdir/final.mat ]; then
   echo Input files $srcdir/final.mdl and/or $srcdir/final.mat do not exist.
   exit 1;
fi

if [ ! -f $graphdir/HCLG.fst -o $graphdir/HCLG.fst -ot $srcdir/final.mdl ]; then
   echo "Graph $graphdir/HCLG.fst does not exist or is too old."
   exit 1;
fi

# We only do one decoding pass, so there is no point caching the
# CMVN stats-- we make them part of a pipe.
feats="ark:compute-cmvn-stats --spk2utt=ark:$data/spk2utt scp:$data/feats.scp ark:- | apply-cmvn --norm-vars=false --utt2spk=ark:$data/utt2spk ark:- scp:$data/feats.scp ark:- | splice-feats ark:- ark:- | transform-feats $srcdir/final.mat ark:- ark:- |"

# For Resource Management, we use beam of 30 and acwt of 1/7.
# More normal, LVCSR setups would have a beam of 13 and acwt of 1/15 or so.
# If you decode with a beam of 20 on an LVCSR setup it will be very slow.

gmm-decode-faster --beam=30.0 --acoustic-scale=0.1429 --word-symbol-table=$lang/words.txt \
  $srcdir/final.mdl $graphdir/HCLG.fst "$feats" ark,t:$dir/test.tra ark,t:$dir/test.ali \
     2> $dir/decode.log || exit 1;

# In this setup there are no non-scored words, so
# scoring is simple.

# the ,p option lets it score partial output without dying..
scripts/sym2int.pl --ignore-first-field $lang/words.txt $data/text | \
  compute-wer --mode=present ark:-  ark,p:$dir/test.tra >& $dir/wer

