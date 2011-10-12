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


# Decoding script that combines two sets of lattices into one.

if [ $# != 5 ]; then
   echo "Usage: steps/decode_combine.sh <data-dir> <lang-dir> <decode-dir-in1> <decode-dir-in2> <decode-dir-in3>"
   echo " e.g.: steps/decode_combine.sh data/test_feb89 data/lang_test exp/tri1/decode/feb89 exp/tri2a/decode/feb89 exp/decode_combine.1.2a/feb89"
   exit 1;
fi

data=$1
lang=$2
indir1=$3
indir2=$4
dir=$5

mkdir -p $dir

if [ -f path.sh ]; then . path.sh; fi


if [[ ! -f $indir1/lat.gz || ! -f $indir2/lat.gz ]]; then
   echo "decode_combine.sh: expecting $indir1/lat.gz and $indir2/lat.gz to both exist."
   exit 1;
fi

lattice-compose "ark:gunzip -c $indir1/lat.gz|" "ark:gunzip -c $indir2/lat.gz|" \
  "ark:|gzip -c > $dir/lat.gz" 2> $dir/compose.log


# Now rescore lattices with various acoustic scales, and compute the WER.
for inv_acwt in 4 5 6 7 8 9 10; do
  acwt=`perl -e "print (1.0/$inv_acwt);"`
  lattice-best-path --acoustic-scale=$acwt --word-symbol-table=$lang/words.txt \
     "ark:gunzip -c $dir/lat.gz|" ark,t:$dir/${inv_acwt}.tra \
     2>$dir/rescore_${inv_acwt}.log

  # Fill in any holes in the transcription due to empty composition.
  scripts/backoff_scp.pl $dir/${inv_acwt}.tra $indir1/${inv_acwt}.tra > $dir/${inv_acwt}_complete.tra

  scripts/sym2int.pl --ignore-first-field $lang/words.txt $data/text | \
   compute-wer --mode=present ark:-  ark,p:$dir/${inv_acwt}_complete.tra \
    >& $dir/wer_${inv_acwt}
done
