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
# [e.g. MFCC] + CMN + LDA + MLLT + SAT features.  It splits the data into
# four chunks and does everything in parallel on the same machine.
# Its output, all in its own
# experimental directory, is {0,1,2,3}.cmvn {0,1,2,3}.ali, {0,1,2,3,}.trans,
# tree, final.mdl, final.mat and final.occs (the last four are just copied
# from the source directory). 


# Option to use precompiled graphs from last phase, if these
# are available (i.e. if they were built with the same data).
# These must be split into four pieces.

oldgraphs=false
if [ "$1" == --use-graphs ]; then
   shift;
   oldgraphs=true # Note: "true" and "false" are the names of the commands
   # "true" and "false", used in bash conditional statements
fi


if [ $# != 4 ]; then
   echo "Usage: steps/align_lda_mllt_sat.sh <data-dir> <lang-dir> <src-dir> <exp-dir>"
   echo " e.g.: steps/align_lda_mllt_sat.sh data/train data/lang exp/tri1 exp/tri1_ali"
   exit 1;
fi

if [ -f path.sh ]; then . path.sh; fi

data=$1
lang=$2
srcdir=$3
dir=$4

oov_sym="<SPOKEN_NOISE>" # Map OOVs to this in training.
grep SPOKEN_NOISE $lang/words.txt >/dev/null || echo "Warning: SPOKEN_NOISE not in dictionary"
silphonelist=`cat $lang/silphones.csl`

mkdir -p $dir
cp $srcdir/{tree,final.mdl,final.alimdl,final.mat,final.occs} $dir || exit 1;  # Create copy of the tree and models and occs...

scale_opts="--transition-scale=1.0 --acoustic-scale=0.1 --self-loop-scale=0.1"

if [ ! -f $data/split4 -o $data/split4 -ot $data/feats.scp ]; then
  scripts/split_data.sh $data 4
fi

echo "Computing cepstral mean and variance statistics"
for n in 0 1 2 3; do
  compute-cmvn-stats --spk2utt=ark:$data/split4/$n/spk2utt scp:$data/split4/$n/feats.scp \
      ark:$dir/$n.cmvn 2>$dir/cmvn$n.log || exit 1;
done


if $oldgraphs; then
  graphdir=$srcdir
  for n in 0 1 2 3; do
   [ ! -f $srcdir/$n.fsts.gz ] && echo You specified --use-graphs but no such file $srcdir/$n.fsts.gz && exit 1;
  done
else
  echo "Compiling training graphs"
  graphdir=$dir
  # If oldgraphs not specified, first create decoding graphs, 
  # since we do two passes of decoding.
  rm $dir/.error 2>/dev/null
  for n in 0 1 2 3; do
    tra="ark:scripts/sym2int.pl --map-oov \"$oov_sym\" --ignore-first-field $lang/words.txt $data/split4/$n/text|";   
    compile-train-graphs $dir/tree $dir/final.mdl  $lang/L.fst "$tra" \
      "ark:|gzip -c >$dir/$n.fsts.gz" 2>$dir/compile_graphs.$n.log || touch $dir/.error &
  done
  wait
  [ -f $dir/.error ] && echo "Error compiling training graphs" && exit 1;
fi


# Align all training data using the supplied model.


rm $dir/.error 2>/dev/null
echo "Aligning data from $data (with alignment model)"

for n in 0 1 2 3; do
  sifeatspart[$n]="ark:apply-cmvn --norm-vars=false --utt2spk=ark:$data/utt2spk ark:$dir/$n.cmvn scp:$data/split4/$n/feats.scp ark:- | splice-feats ark:- ark:- | transform-feats $dir/final.mat ark:- ark:- |"
  featspart[$n]="${sifeatspart[$n]} transform-feats --utt2spk=ark:$data/split4/$n/utt2spk ark:$dir/$n.trans ark:- ark:- |"
done

for n in 0 1 2 3; do
  gmm-align-compiled $scale_opts --beam=10 --retry-beam=40 $dir/final.alimdl \
   "ark:gunzip -c $graphdir/$n.fsts.gz|" "${sifeatspart[$n]}" "ark:|gzip -c >$dir/$n.pre_ali.gz" \
      2> $dir/align_pass1.$n.log || touch $dir/.error &
done
wait;
[ -f $dir/.error ] && echo Error doing pass-1 alignment && exit 1;

echo Computing fMLLR transforms
# Compute fMLLR transforms.
for n in 0 1 2 3; do
 ( ali-to-post "ark:gunzip -c $dir/$n.pre_ali.gz|" ark:- | \
   weight-silence-post 0.0 $silphonelist $dir/final.alimdl ark:- ark:- | \
   gmm-post-to-gpost $dir/final.alimdl "${sifeatspart[$n]}" ark:- ark:- | \
   gmm-est-fmllr-gpost --spk2utt=ark:$data/split4/$n/spk2utt $dir/final.mdl "${sifeatspart[$n]}" \
    ark:- ark:$dir/$n.trans ) 2>$dir/fmllr.$n.log || touch $dir/.error &
done
wait;
[ -f $dir/.error ] && echo Error computing fMLLR transforms && exit 1;

rm $dir/*.pre_ali.gz

echo Doing final alignment
for n in 0 1 2 3; do
  gmm-align-compiled $scale_opts --beam=10 --retry-beam=40 $dir/final.mdl \
   "ark:gunzip -c $graphdir/$n.fsts.gz|" "${featspart[$n]}" "ark:|gzip -c >$dir/$n.ali.gz" \
      2> $dir/align_pass2.$n.log || touch $dir/.error &
done
wait;
[ -f $dir/.error ] && echo Error doing pass-2 alignment && exit 1;

rm $dir/*.fsts.gz 2>/dev/null; # In case we made graphs in this directory.

echo "Done aligning data."
