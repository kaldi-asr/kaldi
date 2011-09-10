#!/bin/bash
# Copyright 2011 Microsoft Corporation1 Gilles Boulianne

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



if [ $# != 4 ]; then
   echo "Usage: scripts/latoracle.sh <lattice-rspecifier> <transcript-text-file> <output-decode-dir> <param>"
   exit 1;
fi

. path.sh || exit 1;


inputlat=$1   # e.g. "ark:gunzip -c /pub/tmp/kaldi2011/dpovey/decode_tri1_latgen/test_sep92.lat.gz|"
transcript=$2 # e.g. data_prep/test_sep92_trans.txt
dir=$3        # e.g. exp/decode_tri1_latgen
param=$4      # ouput files will be given "param" suffix as in wer_${param}

mkdir -p $dir

# Create reference transcriptions and lattices

cat $transcript | sed 's:<NOISE>::g' |  sed 's:<SPOKEN_NOISE>::g' > $dir/test_trans.filt
cat $dir/test_trans.filt | \
  scripts/sym2int.pl --ignore-first-field data/words.txt | \
  string-to-lattice "ark:$dir/test_trans.lats" 2>$dir/reference.${param}.log

lattice-oracle --word-symbol-table=data/words.txt \
     "ark:$dir/test_trans.lats" "ark:gunzip -c $dir/lats.pruned.gz|" "ark,t:$dir/oracle_${param}.tra"  \
        2>$dir/oracle.${param}.log
  
# the ,p option lets it score partial output without dying..
cat $dir/oracle_${param}.tra | \
scripts/int2sym.pl --ignore-first-field data/words.txt | \
sed 's:<s>::' | sed 's:</s>::' | sed 's:<UNK>::g' | \
compute-wer --text --mode=present ark:$dir/test_trans.filt  ark,p:- >& $dir/wer_${param}


