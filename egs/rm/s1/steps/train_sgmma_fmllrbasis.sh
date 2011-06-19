#!/bin/bash

# Copyright 2010-2011	Saarland University
# Author: Arnab Ghoshal

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

# Training the bases for fMLLR. This expects exp/sgmma/final_fmllr.mdl with the
# fMLLR pre-transforms to be already present. If not, it will look for 
# exp/sgmma/final.mdl and compute the fMLLR pre-transforms. 
# The output is exp/sgmma/final_fmllrbasis.mdl with the bases & pre-transforms.

if [ -f path.sh ]; then . path.sh; fi

dir=exp/sgmma
imodel=$dir/final.mdl
occs=exp/sgmma/final.occs
model=$dir/final_fmllr.mdl
omodel=$dir/final_fmllrbasis.mdl

silphonelist=`cat data/silphones.csl`
silweight=0.0

nbases=200

if [ ! -f $model ]; then
  if [ ! -f $imodel ] || [ ! -f $occs ]; then
    echo "Missing model ($imodel) or occs ($occs). Maybe training didn't finish?"
    exit 1;
  fi
  sgmm-comp-prexform $imodel $occs $model
fi

feats="ark:add-deltas --print-args=false scp:data/train.scp ark:- |"

if [ ! -f $dir/gselect.gz ]; then
  sgmm-gselect $model "$feats" ark,t:- 2>$dir/gselect.log | gzip -c \
      > $dir/gselect.gz || exit 1;
fi
gselect_opt="--gselect=ark:gunzip -c $dir/gselect.gz|"

scale_opts="--transition-scale=1.0 --acoustic-scale=0.1 --self-loop-scale=0.1"
echo "Aligning data"
sgmm-align-compiled $spkvecs_opt $scale_opts "$gselect_opt" --beam=8 \
  --retry-beam=40 $model "ark:gunzip -c $dir/graphs.fsts.gz|" "$feats" \
   ark:$dir/final.ali 2> $dir/align.final.log || exit 1;

spk2utt="ark:data/train.spk2utt"
echo "Accumulating stats for fMLLR bases."

sgmm-acc-fmllrbasis-ali --verbose=3 "$gselect_opt" --binary=true \
    --sil-phone-list=$silphonelist --sil-weight=$silweight \
    $model "$feats" ark:$dir/final.ali $spk2utt $dir/fmllrbasis.acc \
    2> $dir/acc.fmllrbasis.log || exit 1;

echo "Estimating fMLLR bases."
sgmm-est-fmllrbasis --verbose=3 --num-bases=$nbases $model $omodel \
  $dir/fmllrbasis.acc 2> $dir/update.fmllrbasis.log || exit 1;
