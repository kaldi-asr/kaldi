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



if [ $# != 3 ]; then
   echo "Usage: scripts/latoracle.sh <input-decode-dir> <transcript-text-file> <output-decode-dir>"
   exit 1;
fi

. path.sh || exit 1;


inputdir=$1  # e.g. /pub/tmp/kaldi2011/dpovey/decode_tri1_latgen/
transcript=$2 # e.g. data_prep/test_sep92_trans.txt
dir=$3  #eg exp/decode_tri1_latgen

mkdir -p $dir

# Create reference transcriptions and lattices
cat $transcript | sed 's:<NOISE>::g' |  sed 's:<SPOKEN_NOISE>::g' > $dir/test_trans.filt
cat $dir/test_trans.filt | \
  scripts/sym2int.pl --ignore-first-field data/words.txt | \
  string-to-lattice "ark:$dir/test_trans.lats" 2>$dir/string-to-lattice.log

# Symbols that don't count as errors
echo "<s>"    > $dir/ignore.txt
echo "</s>"  >> $dir/ignore.txt
echo "<UNK>" >> $dir/ignore.txt

# Loop over pruning beams
inv_acwt=10
acwt=`perl -e "print (1.0/$inv_acwt);"`
for beam in 0.01 1 5 10; do

  echo "Pruning $inputdir"'/*.lats.gz'" with invacwt=$inv_acwt and beam=$beam"
  lattice-prune --acoustic-scale=$acwt --beam=$beam \
    "ark:gunzip -c $inputdir/*.lats.gz|" "ark,t:|gzip -c>$dir/lats.pruned.gz" \
       2>$dir/prune.$beam.log

  echo "Computing oracle error rate w/r $transcript"
  lattice-oracle --word-symbol-table=data/words.txt --wildcard-symbols-list=$dir/ignore.txt \
     "ark:$dir/test_trans.lats" "ark:gunzip -c $dir/lats.pruned.gz|" "ark,t:$dir/oracle${beam}.tra" \
       2>$dir/oracle.$beam.log
  
  cat $dir/oracle${beam}.tra | \
   scripts/int2sym.pl --ignore-first-field data/words.txt | \
   sed 's:<s>::' | sed 's:</s>::' | sed 's:<UNK>::g' | \
    compute-wer --text --mode=present ark:$dir/test_trans.filt  ark,p:- | tee $dir/wer_${beam}
done

