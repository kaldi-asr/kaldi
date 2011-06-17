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

# Train separate male and female UBMs and merge them
# to get a single UBM with known male and female indices.

if [ -f path.sh ]; then . path.sh; fi

dir=exp/ubm3b
mkdir -p $dir
srcdir=exp/tri1

init-ubm --intermediate-numcomps=2000 --ubm-numcomps=400 --verbose=2 \
    --fullcov-ubm=true $srcdir/final.mdl $srcdir/final.occs \
    $dir/0m.ubm 2> $dir/cluster.log

cp $dir/0.m.ubm $dir/0.f.ubm


# Use just SI-84 data, split into male and female.

head -7138 data/train.scp  > $dir/train.scp

scripts/compose_maps.pl data/train.utt2spk data/spk2gender.map | grep -w m | \
   awk '{print $1}' | scripts/filter_scp.pl - $dir/train.scp > $dir/train_m.scp
scripts/compose_maps.pl data/train.utt2spk data/spk2gender.map | grep -w f | \
   awk '{print $1}' | scripts/filter_scp.pl - $dir/train.scp > $dir/train_f.scp

scripts/split_scp.pl $dir/train_m.scp  $dir/train_m{1,2}.scp
scripts/split_scp.pl $dir/train_f.scp  $dir/train_f{1,2}.scp

rm -f $dir/.error

for x in 0 1 2 3; do
    echo "Pass $x"
    for n in 1 2; do
      for g in m f; do
        feats="ark:add-deltas scp:$dir/train_${g}${n}.scp ark:- |"
        fgmm-global-acc-stats --diag-gmm-nbest=15 --binary=false --verbose=2 $dir/$x.$g.ubm "$feats" \
          $dir/$x.$g.$n.acc 2> $dir/acc.$x.$g.$n.log  || touch $dir/.error &
      done
    done
    wait;
    [ -f $dir/.error ] && echo "Error accumulating stats" && exit 1;
    for g in m f; do
     ( fgmm-global-sum-accs - $dir/$x.$g.{1,2}.acc | \
       fgmm-global-est --verbose=2 $dir/$x.${g}.ubm - $dir/$[$x+1].${g}.ubm ) 2> $dir/update.$x.$g.log || exit 1;
      rm $dir/$x.$g.{1,2}.acc $dir/$x.${g}.ubm
    done
done

rm $dir/final.?.ubm 2>/dev/null
ln -s 4.m.ubm $dir/final.m.ubm
ln -s 4.f.ubm $dir/final.f.ubm

fgmm-global-merge $dir/final.ubm $dir/sizes $dir/final.m.ubm $dir/final.f.ubm 2> $dir/merge.log || exit 1;

# Now create the preselect lists for male and female
# Approximately, the male one is the numbers 0..399 and the female one is the numbers
# 400..799.
# [it's actually slightly different due to pruning; the male one has only 399 not 400
#  indices.]
cat $dir/sizes | awk '{ printf "m "; for (x=0; x < $1; x++) { printf("%d ", x); } print ""; 
                       printf "f "; for (x=$1; x < $1+$2; x++) { printf("%d ", x); } print ""; }' > $dir/preselect.map

