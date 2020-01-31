#!/usr/bin/env bash
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

if [ -f ./path.sh ]; then . ./path.sh; fi

if [ $# -ne 3 ]; then
   echo "Usage: score_lats.sh <decode-dir> <word-symbol-table> <data-dir>"
   exit 1;
fi

dir=$1
symtab=$2
data=$3

if [ ! -f $symtab ]; then
  echo No such word symbol table file $symtab
  exit 1;
fi
if [ ! -f $data/text ]; then
  echo Could not find transcriptions in $data/text
  exit 1
fi


trans=$data/text

cat $trans | sed 's:<NOISE>::g' |  sed 's:<SPOKEN_NOISE>::g' > $dir/test_trans.filt

for inv_acwt in 9 10 11 12 13 14 15 16 17 18 19 20; do 
   acwt=`perl -e "print (1.0/$inv_acwt);"`
   lattice-best-path --acoustic-scale=$acwt --word-symbol-table=$symtab \
      "ark:gunzip -c $dir/lat.*.gz|" ark,t:$dir/${inv_acwt}.tra \
      2>$dir/rescore_${inv_acwt}.log
     
   cat $dir/${inv_acwt}.tra | \
    int2sym.pl --ignore-first-field $symtab | sed 's:<UNK>::g' | \
    compute-wer --text --mode=present ark:$dir/test_trans.filt  ark,p:-   >& $dir/wer_$inv_acwt
done

