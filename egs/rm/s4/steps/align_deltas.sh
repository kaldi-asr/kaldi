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

# This script does training-data alignment given a model built using 
# CMN + delta + delta-delta features.  Its output, all in its own
# experimental directory, is cmvn.ark, ali, tree, and final.mdl 
# (the last two are just copied from the source directory). 

# Option to use precompiled graphs from last phase, if these
# are available (i.e. if they were built with the same data).

graphs=
if [ "$1" == --graphs ]; then
   shift;
   graphs=$1
   shift
fi


if [ $# != 4 ]; then
   echo "Usage: steps/align_deltas.sh <data-dir> <lang-dir> <src-dir> <exp-dir>"
   echo " e.g.: steps/align_deltas.sh data/train data/lang exp/tri1 exp/tri1_ali"
   exit 1;
fi

if [ -f path.sh ]; then . path.sh; fi

data=$1
lang=$2
srcdir=$3
dir=$4



mkdir -p $dir
cp $srcdir/{tree,final.mdl,final.occs} $dir || exit 1;  # Create copy of the tree and model and occs...

scale_opts="--transition-scale=1.0 --acoustic-scale=0.1 --self-loop-scale=0.1"



echo "Computing cepstral mean and variance statistics"
compute-cmvn-stats scp:$data/feats.scp \
     ark:$dir/cmvn.ark 2>$dir/cmvn.log || exit 1;

feats="ark:apply-cmvn --norm-vars=false ark:$dir/cmvn.ark scp:$data/feats.scp ark:- | add-deltas ark:- ark:- |"

# Align all training data using the supplied model.

echo "Aligning all training data"
if [ -z "$graphs" ]; then # --graphs option not supplied [-z means empty string]
  # compute integer form of transcripts.
  scripts/sym2int.pl --ignore-first-field $lang/words.txt < $data/text > $dir/train.tra \
    || exit 1;
  gmm-align $scale_opts --beam=8 --retry-beam=40 $dir/tree $dir/final.mdl $lang/L.fst \
   "$feats" ark:$dir/train.tra ark:$dir/ali 2> $dir/align.log || exit 1;
  rm $dir/train.tra
else
  gmm-align-compiled $scale_opts --beam=8 --retry-beam=40 $dir/final.mdl \
   "$graphs" "$feats" ark:$dir/ali 2> $dir/align.log || exit 1;
fi

echo "Done."
