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
# Its output, all in its own experimental directory, is (assuming
# you don't change the #jobs with --num-job option),
# {0,1,2,3}.cmvn {0,1,2,3}.ali.gz, {0,1,2,3}.trans, tree, final.mdl ,
# final.mat and final.occs (the last four are just copied from the source directory). 


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
     [ "$cmd" == "" ] && echo "Empty string given to --cmd option" && exit 1;
     shift
  fi  
done

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

oov_sym=`cat $lang/oov.txt`
silphonelist=`cat $lang/silphones.csl`

mkdir -p $dir
cp $srcdir/{tree,final.mdl,final.alimdl,final.mat,final.occs} $dir || exit 1;  # Create copy of the tree and models and occs...

scale_opts="--transition-scale=1.0 --acoustic-scale=0.1 --self-loop-scale=0.1"

if [ ! -d $data/split$nj -o $data/split$nj -ot $data/feats.scp ]; then
  scripts/split_data.sh $data $nj
fi

echo "Computing cepstral mean and variance statistics"
for n in `get_splits.pl $nj`; do
  compute-cmvn-stats --spk2utt=ark:$data/split$nj/$n/spk2utt scp:$data/split$nj/$n/feats.scp \
      ark:$dir/$n.cmvn 2>$dir/cmvn$n.log || exit 1;
done


if $oldgraphs; then
  graphdir=$srcdir
  for n in `get_splits.pl $nj`; do
   [ ! -f $srcdir/$n.fsts.gz ] && echo You specified --use-graphs but no such file $srcdir/$n.fsts.gz && exit 1;
  done
else
  echo "Compiling training graphs"
  graphdir=$dir
  # If oldgraphs not specified, first create decoding graphs, 
  # since we do two passes of decoding.
  rm $dir/.error 2>/dev/null
  for n in `get_splits.pl $nj`; do
    tra="ark:scripts/sym2int.pl --map-oov \"$oov_sym\" --ignore-first-field $lang/words.txt $data/split$nj/$n/text|";   
    $cmd $dir/compile_graphs.$n.log  \
      compile-train-graphs $dir/tree $dir/final.mdl  $lang/L.fst "$tra" \
        "ark:|gzip -c >$dir/$n.fsts.gz" || touch $dir/.error &
  done
  wait
  [ -f $dir/.error ] && echo "Error compiling training graphs" && exit 1;
fi


# Align all training data using the supplied model.


rm $dir/.error 2>/dev/null
echo "Aligning data from $data (with alignment model)"

for n in `get_splits.pl $nj`; do
  sifeatspart[$n]="ark,s,cs:apply-cmvn --norm-vars=false --utt2spk=ark:$data/utt2spk ark:$dir/$n.cmvn scp:$data/split$nj/$n/feats.scp ark:- | splice-feats ark:- ark:- | transform-feats $dir/final.mat ark:- ark:- |"
  featspart[$n]="${sifeatspart[$n]} transform-feats --utt2spk=ark:$data/split$nj/$n/utt2spk ark:$dir/$n.trans ark:- ark:- |"
done

for n in `get_splits.pl $nj`; do
  $cmd $dir/align_pass1.$n.log \
  gmm-align-compiled $scale_opts --beam=10 --retry-beam=40 $dir/final.alimdl \
   "ark:gunzip -c $graphdir/$n.fsts.gz|" "${sifeatspart[$n]}" "ark:|gzip -c >$dir/$n.pre_ali.gz" \
     || touch $dir/.error &
done
wait;
[ -f $dir/.error ] && echo Error doing pass-1 alignment && exit 1;

echo Computing fMLLR transforms
# Compute fMLLR transforms.
for n in `get_splits.pl $nj`; do
  $cmd $dir/fmllr.$n.log \
    ali-to-post "ark:gunzip -c $dir/$n.pre_ali.gz|" ark:- \| \
      weight-silence-post 0.0 $silphonelist $dir/final.alimdl ark:- ark:- \| \
      gmm-post-to-gpost $dir/final.alimdl "${sifeatspart[$n]}" ark:- ark:- \| \
      gmm-est-fmllr-gpost --spk2utt=ark:$data/split$nj/$n/spk2utt $dir/final.mdl "${sifeatspart[$n]}" \
        ark,s,cs:- ark:$dir/$n.trans || touch $dir/.error &
done
wait;
[ -f $dir/.error ] && echo Error computing fMLLR transforms && exit 1;

rm $dir/*.pre_ali.gz

echo Doing final alignment
for n in `get_splits.pl $nj`; do
  $cmd $dir/align_pass2.$n.log \
    gmm-align-compiled $scale_opts --beam=10 --retry-beam=40 $dir/final.mdl \
      "ark:gunzip -c $graphdir/$n.fsts.gz|" "${featspart[$n]}" "ark:|gzip -c >$dir/$n.ali.gz" \
       || touch $dir/.error &
done
wait;
[ -f $dir/.error ] && echo Error doing pass-2 alignment && exit 1;

rm $dir/*.fsts.gz 2>/dev/null; # In case we made graphs in this directory.

echo "Done aligning data."
