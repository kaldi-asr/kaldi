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
# [e.g. MFCC] + CMN + LDA + MLLT features.  It splits the data into
# four chunks and does everything in parallel on the same machine.
# Its output, all in its own experimental directory, is (assuming
# you don't change the #jobs with --num-job option),
# {0,1,2,3}.cmvn {0,1,2,3}.ali.gz, tree, final.mdl ,
# final.mat and final.occs (the last four are just copied from the source directory). 


# Option to use precompiled graphs from last phase, if these
# are available (i.e. if they were built with the same data).
# These must be split into four pieces.

nj=4
cmd=scripts/run.pl
oldgraphs=false
for x in 1 2 3; do
  if [ "$1" == --use-graphs ]; then
    shift;
    oldgraphs=true
  fi
  if [ $1 == "--num-jobs" ]; then
     shift
     nj=$1
     shift
  fi
  if [ $1 == "--cmd" ]; then
     shift
     cmd=$1
     shift
  fi  
done


if [ $# != 4 ]; then
   echo "Usage: steps/align_lda_mllt.sh <data-dir> <lang-dir> <src-dir> <exp-dir>"
   echo " e.g.: steps/align_lda_mllt.sh data/train data/lang exp/tri1 exp/tri1_ali"
   exit 1;
fi

if [ -f path.sh ]; then . path.sh; fi

data=$1
lang=$2
srcdir=$3
dir=$4

oov_sym=`cat $lang/oov.txt`


mkdir -p $dir
cp $srcdir/{tree,final.mdl,final.mat,final.occs} $dir || exit 1;  # Create copy of the tree and model and occs...

scale_opts="--transition-scale=1.0 --acoustic-scale=0.1 --self-loop-scale=0.1"

if [ ! -d $data/split$nj -o $data/split$nj -ot $data/feats.scp ]; then
  split_data.sh $data $nj
fi

echo "Computing cepstral mean and variance statistics"
for n in `get_splits.pl $nj`; do
  compute-cmvn-stats --spk2utt=ark:$data/split$nj/$n/spk2utt scp:$data/split$nj/$n/feats.scp \
      ark:$dir/$n.cmvn 2>$dir/cmvn$n.log || exit 1;
done


# Align all training data using the supplied model.


rm $dir/.error 2>/dev/null
echo "Aligning data from $data"
if $oldgraphs; then 
  for n in `get_splits.pl $nj`; do
    feats="ark:apply-cmvn --norm-vars=false --utt2spk=ark:$data/utt2spk ark:$dir/$n.cmvn scp:$data/split$nj/$n/feats.scp ark:- | splice-feats ark:- ark:- | transform-feats $dir/final.mat ark:- ark:- |"
    if [ ! -f $srcdir/$n.fsts.gz ]; then
       echo You specified --use-graphs but no such file $srcdir/$n.fsts.gz
       exit 1;
    fi
    $cmd $dir/align$n.log \
      gmm-align-compiled $scale_opts --beam=10 --retry-beam=40 $dir/final.mdl \
       "ark:gunzip -c $srcdir/$n.fsts.gz|" "$feats" "ark:|gzip -c >$dir/$n.ali.gz" \
       || touch $dir/.error &
  done
  wait;
  [ -f $dir/.error ] && echo error doing alignment && exit 1;
else
  for n in `get_splits.pl $nj`; do
    feats="ark,s,cs:apply-cmvn --norm-vars=false --utt2spk=ark:$data/utt2spk ark:$dir/$n.cmvn scp:$data/split$nj/$n/feats.scp ark:- | splice-feats ark:- ark:- | transform-feats $dir/final.mat ark:- ark:- |"
    # compute integer form of transcripts.
    tra="ark:sym2int.pl --map-oov \"$oov_sym\" --ignore-first-field $lang/words.txt $data/split$nj/$n/text|";
    $cmd $dir/align$n.log \
     compile-train-graphs $dir/tree $dir/final.mdl  $lang/L.fst "$tra" ark:- \| \
       gmm-align-compiled $scale_opts --beam=10 --retry-beam=40 $dir/final.mdl ark:- \
        "$feats" "ark:|gzip -c >$dir/$n.ali.gz" || touch $dir/.error &
  done
  wait;
  [ -f $dir/.error ] && echo error doing alignment && exit 1;
fi

#Also merge alignments to single archive (for nnet training)
gunzip -c $dir/*.ali.gz | gzip -c > $dir/ali.gz

echo "Done aligning data."
