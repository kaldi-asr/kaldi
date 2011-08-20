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
   echo "Usage: steps/decode_lda_et.sh <model-dir> <data-dir> <lang-dir> <decode-dir>"
   echo " e.g.: steps/decode_lda_et.sh exp/tri2c data/test_feb89 data/lang_test exp/tri2c/decode_feb89"
   exit 1;
fi

srcdir=$1
data=$2
lang=$3
dir=$4
graphdir=$srcdir/graph

silphonelist=`cat $lang/silphones.csl`

mkdir -p $dir

if [ -f path.sh ]; then . path.sh; fi

# -f means file exists; -o means or.
requirements="$srcdir/final.mdl $srcdir/final.alimdl $srcdir/final.mat $srcdir/final.et"

for f in $requirements; do
  if [ ! -f $f ]; then
    echo "decode_lda_et.sh: input file $f does not exist";
    exit 1;
  fi
done

if [ ! -f $graphdir/HCLG.fst -o $graphdir/HCLG.fst -ot $srcdir/final.mdl ]; then
   echo "Graph $graphdir/HCLG.fst does not exist or is too old."
   exit 1;
fi

# Compute CMVN stats.
compute-cmvn-stats --spk2utt=ark:$data/spk2utt scp:$data/feats.scp ark,t:$dir/cmvn.ark \
   2>$dir/cmvn.log

sifeats="ark:apply-cmvn --norm-vars=false --utt2spk=ark:$data/utt2spk ark:$dir/cmvn.ark scp:$data/feats.scp ark:- | splice-feats ark:- ark:- | transform-feats $srcdir/final.mat ark:- ark:- |"

# For Resource Management, we use beam of 30 and acwt of 1/7.
# More normal, LVCSR setups would have a beam of 13 and acwt of 1/15 or so.
# If you decode with a beam of 20 on an LVCSR setup it will be very slow.

gmm-decode-faster --beam=30.0 --acoustic-scale=0.1429 --word-symbol-table=$lang/words.txt \
  $srcdir/final.alimdl $graphdir/HCLG.fst "$sifeats" ark,t:$dir/pass1.tra ark,t:$dir/pass1.ali \
     2> $dir/decode_pass1.log || exit 1;

( ali-to-post ark:$dir/pass1.ali ark:- | \
   weight-silence-post 0.0 $silphonelist $srcdir/final.alimdl ark:- ark:- | \
   gmm-post-to-gpost $srcdir/final.alimdl "$sifeats" ark:- ark:- | \
   gmm-est-et --spk2utt=ark:$data/spk2utt $srcdir/final.mdl $srcdir/final.et "$sifeats" \
       ark,s,cs:- ark:$dir/trans.ark ark,t:$dir/warp ) \
     2> $dir/trans.log || exit 1;

feats="ark:apply-cmvn --norm-vars=false --utt2spk=ark:$data/utt2spk ark:$dir/cmvn.ark scp:$data/feats.scp ark:- | splice-feats ark:- ark:- | transform-feats $srcdir/final.mat ark:- ark:- | transform-feats --utt2spk=ark:$data/utt2spk ark:$dir/trans.ark ark:- ark:- |"

# Second pass decoding...
gmm-decode-faster --beam=30.0 --acoustic-scale=0.1429 --word-symbol-table=$lang/words.txt \
  $srcdir/final.mdl $graphdir/HCLG.fst "$feats" ark,t:$dir/pass2.tra ark,t:$dir/pass2.ali \
     2> $dir/decode_pass2.log || exit 1;


# In this setup there are no non-scored words, so
# scoring is simple.

# the ,p option lets it score partial output without dying..
scripts/sym2int.pl --ignore-first-field $lang/words.txt $data/text | \
  compute-wer --mode=present ark:-  ark,p:$dir/pass2.tra >& $dir/wer

