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


# To view the lattices, a suitable command (after running this) is:
# gunzip -c exp/decode_tri1_latgen/test_feb89.lat.gz | scripts/int2sym.pl --field 3 data/words.txt | less

if [ -f path.sh ]; then . path.sh; fi

beamstotry="0.01 0.5 1 5 10"

inputdir=exp/decode_tri1_latgen   # default value

if [ $# == 1 ]; then
  inputdir=$1;
fi

dir=exp/decode_tri1_latoracle

mkdir -p $dir

inv_acwt=10
acwt=`perl -e "print (1.0/$inv_acwt);"`

for test in mar87 oct87 feb89 oct89 feb91 sep92; do

  inputlat="ark:gunzip -c $inputdir/test_${test}.lat.gz|"

  # try pruning beams
  for beam in $beamstotry; do

    echo "Pruning lattices $inputlat with invacwt=$inv_acwt and beam=$beam"
    lattice-prune --acoustic-scale=$acwt --beam=$beam \
      "$inputlat" "ark,t:|gzip -c>$dir/lats.pruned.gz"  \
           2>$dir/prune.$beam.log

    scripts/latoracle.sh "ark:gunzip -c $dir/lats.pruned.gz|" data_prep/test_${test}_trans.txt $dir ${test}_${beam}
  done

done

for beam in $beamstotry; do
 echo -n "Beam $beam "
 grep WER $dir/wer_{mar87,oct87,feb89,oct89,feb91,sep92}_${beam} | \
  awk '{n=n+$4; d=d+$6} END{ printf("Average WER is %f (%d / %d) \n", 100.0*n/d, n, d); }' \
   | tee $dir/wer_${beam}
done

