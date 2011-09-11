#!/bin/bash
# Copyright 2010-2011 Microsoft Corporation  Arnab Ghoshal

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

# To be run from ..

# This script does training-data alignment given a model built using CMN +
# splice-9-frames + LDA + ET features.  Its output, all in
# its own experimental directory, is cmvn.ark, trans.ark, ali, tree, final.mdl,
# final.alimdl, final.occs, final.mat and final.et (the last six are just copied
# from the source directory).

# Option to use precompiled graphs from last phase, if these
# are available (i.e. if they were built with the same data).

graphs=
if [ "$1" == --graphs ]; then
   shift;
   graphs=$1
   shift
fi


if [ $# != 4 ]; then
   echo "Usage: steps/align_lda_et.sh <data-dir> <lang-dir> <src-dir> <exp-dir>"
   echo " e.g.: steps/align_lda_et.sh data/train data/lang exp/tri2c exp/tri2c_ali"
   exit 1;
fi

if [ -f path.sh ]; then . path.sh; fi

data=$1
lang=$2
srcdir=$3
dir=$4

requirements="$srcdir/final.mdl $srcdir/final.alimdl $srcdir/final.mat $srcdir/final.et $srcdir/tree"
for f in $requirements; do
  if [ ! -f $f ]; then
     echo "align_lda_mllt.sh: no such file $f"
     exit 1;
  fi
done

mkdir -p $dir
cp $srcdir/{final.mdl,final.occs,final.alimdl,tree,final.mat,final.et} $dir || exit 1;  # Create copies in $dir

scale_opts="--transition-scale=1.0 --acoustic-scale=0.1 --self-loop-scale=0.1"

echo "Computing cepstral mean and variance statistics"
compute-cmvn-stats --spk2utt=ark:$data/spk2utt scp:$data/feats.scp \
     ark:$dir/cmvn.ark 2>$dir/cmvn.log || exit 1;

sifeats="ark:apply-cmvn --norm-vars=false --utt2spk=ark:$data/utt2spk ark:$dir/cmvn.ark scp:$data/feats.scp ark:- | splice-feats ark:- ark:- | transform-feats $dir/final.mat ark:- ark:- |"

# Align all training data using the supplied alignment model.

echo "Aligning all training data [with alignment model]"
if [ -z "$graphs" ]; then # --graphs option not supplied [-z means empty string]
  # compute integer form of transcripts.
  scripts/sym2int.pl --ignore-first-field $lang/words.txt < $data/text > $dir/train.tra \
    || exit 1;
  gmm-align $scale_opts --beam=8 --retry-beam=40 $dir/tree $dir/final.alimdl $lang/L.fst \
   "$sifeats" ark:$dir/train.tra ark:$dir/pre.ali 2> $dir/align_pass1.log || exit 1;
  rm $dir/train.tra
else
  gmm-align-compiled $scale_opts --beam=8 --retry-beam=40 $dir/final.alimdl  \
   "$graphs" "$sifeats" ark:$dir/pre.ali 2> $dir/align_pass1.log || exit 1;
fi

echo "Computing exponential transforms"

silphonelist=`cat $lang/silphones.csl`
[ -z $silphonelist ] && exit 1;
( ali-to-post ark:$dir/pre.ali ark:- | \
  weight-silence-post 0.0 $silphonelist $dir/final.alimdl ark:- ark:- | \
  gmm-post-to-gpost $dir/final.alimdl "$sifeats" ark:- ark:- | \
  gmm-est-et --spk2utt=ark:$data/spk2utt $dir/final.mdl $dir/final.et \
    "$sifeats" ark,s,cs:- ark:$dir/trans.ark ark,t:$dir/warp ) \
    2>$dir/trans.log || exit 1;

feats="$sifeats transform-feats --utt2spk=ark:$data/utt2spk ark:$dir/trans.ark ark:- ark:- |"

echo "Aligning all training data [with final model and features]"
if [ -z "$graphs" ]; then # --graphs option not supplied [-z means empty string]
  # compute integer form of transcripts.
  scripts/sym2int.pl --ignore-first-field $lang/words.txt < $data/text > $dir/train.tra \
    || exit 1;
  gmm-align $scale_opts --beam=8 --retry-beam=40 $dir/tree $dir/final.mdl $lang/L.fst \
   "$feats" ark:$dir/train.tra ark:$dir/ali 2> $dir/align_pass2.log || exit 1;
  rm $dir/train.tra
else
  gmm-align-compiled $scale_opts --beam=8 --retry-beam=40 $dir/final.mdl  \
   "$graphs" "$feats" ark:$dir/ali 2> $dir/align_pass2.log || exit 1;
fi

rm $dir/pre.ali

echo "Done."
