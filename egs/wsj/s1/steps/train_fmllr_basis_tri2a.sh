#!/bin/bash
# Copyright 2012   Carnegie Mellon University  Yajie Miao

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


# Based on the system in tri2a, this script trains fMLLR base matrices,
# which are used in basis-fMLLR adaptation.  
# To enable parallelization, we need to split scp based on speaker boundaries.
# Therefore, we need to recompile training graphs and regenate alignment on 
# the new splits.
# Accumulation can also be conducted in a per-utterance mode (the commented part).
# But we haven't verified this manner.

if [ -f path.sh ]; then . path.sh; fi

dir=exp/tri2a_fmllr_basis
srcdir=exp/tri2a
srcmodel=$srcdir/final.mdl
scale_opts="--transition-scale=1.0 --acoustic-scale=0.1 --self-loop-scale=0.1"

silphonelist=`cat data/silphones.csl`

mkdir -p $dir
cp $srcdir/train.scp $dir
cp $srcdir/train.tra $dir

# Parallelization on 3 cpus; split on speaker boundaries
scripts/filter_scp.pl $dir/train.scp data/train.utt2spk > $dir/train.utt2spk

scripts/split_scp.pl --utt2spk=$dir/train.utt2spk $dir/train{,1,2,3}.scp
scripts/split_scp.pl --utt2spk=$dir/train.utt2spk $dir/train{,1,2,3}.tra
scripts/split_scp.pl --utt2spk=$dir/train.utt2spk $dir/train{,1,2,3}.utt2spk

for n in 1 2 3 ""; do # The "" handles the un-split one.  Creating spk2utt files..
  scripts/utt2spk_to_spk2utt.pl $dir/train$n.utt2spk > $dir/train$n.spk2utt
done

# also see featspart below, used for sub-parts of the features;
# try to keep them in sync.
feats="ark:add-deltas --print-args=false scp:$dir/train.scp ark:- |"
for n in 1 2 3; do
   featspart[$n]="ark:add-deltas --print-args=false scp:$dir/train${n}.scp ark:- |"
done

# Recompile graphs since we have different data splits as train_tri2a.
# Align all training data using old model.

echo "Aligning all training data"

rm -f $dir/.error
for n in 1 2 3; do
  compile-train-graphs $srcdir/tree $srcmodel data/L.fst ark:$dir/train${n}.tra ark:- 2>$dir/graphsold.${n}.log | \
   gmm-align-compiled $scale_opts --beam=8 --retry-beam=40 $srcmodel ark:- "${featspart[$n]}" "ark:|gzip -c > $dir/${n}.ali.gz" 2> $dir/align.0.${n}.log || touch $dir/.error &
done
wait;
[ -f $dir/.error ] &&  echo compile-graphs error RE old system && exit 1

# Accumulation over all training speakers
rm -f $dir/.error
for n in 1 2 3; do
( ali-to-post "ark:gunzip -c $dir/${n}.ali.gz|" ark:- | \
  weight-silence-post 0.0 $silphonelist $srcmodel ark:- ark:- | \
  gmm-basis-fmllr-accs --spk2utt=ark:$dir/train$n.spk2utt $srcmodel \
  "${featspart[$n]}" ark,o:- $dir/$n.basis.acc ) \
  2> $dir/basis.acc.$n.log || touch $dir/.error &
done
wait
[ -f $dir/.error ] &&  echo error accumulating basis fMLLR stats && exit 1

# Accumulation over all training utterances
#rm -f $dir/.error
#for n in 1 2 3; do
#( ali-to-post "ark:gunzip -c $dir/${n}.ali.gz|" ark:- | \
#  weight-silence-post 0.0 $silphonelist $srcmodel ark:- ark:- | \
#  gmm-basis-fmllr-accs $srcmodel \
#  "${featspart[$n]}" ark,o:- $dir/$n.basis.acc ) \
#  2> $dir/basis.acc.$n.log || touch $dir/.error &
#done
#wait
#[ -f $dir/.error ] &&  echo error accumulating basis fMLLR stats && exit 1

# Estimate base matrices
gmm-basis-fmllr-training $srcmodel $dir/fmllr.base.mats $dir/{1,2,3}.basis.acc


