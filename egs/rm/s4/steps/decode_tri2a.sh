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

# to be run from ..

echo "--- Starting TRI2a decoding ..."

if [ -f path.sh ]; then . path.sh; fi
dir=exp/decode_tri2a
mkdir -p $dir
model=exp/tri2a/final.mdl
tree=exp/tri2a/tree
graphdir=exp/graph_tri2a

scripts/mkgraph.sh $tree $model $graphdir

feats="ark:add-deltas --print-args=false scp:data/test.scp ark:- |"

gmm-decode-faster --beam=20.0 --acoustic-scale=0.08333 --word-symbol-table=data/words.txt $model $graphdir/HCLG.fst "$feats" ark,t:$dir/test.tra ark,t:$dir/test.ali  2> $dir/decode.log

# the ,p option lets it score partial output without dying..
scripts/sym2int.pl --ignore-first-field data/words.txt data_prep/test_trans.txt | \
compute-wer --mode=present ark:-  ark,p:$dir/test.tra >& $dir/wer

echo "--- Done TRI2a decoding!"
