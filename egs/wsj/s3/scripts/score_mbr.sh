#!/bin/bash

# Script for minimum bayes risk decoding.

if [ -f ./path.sh ]; then . ./path.sh; fi

if [ $# -ne 3 ]; then
   echo "Usage: scripts/score_mbr.sh <decode-dir> <word-symbol-table> <data-dir>"
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
   lattice-mbr-decode --acoustic-scale=$acwt --word-symbol-table=$symtab \
      "ark:gunzip -c $dir/lat.*.gz|" ark,t:$dir/${inv_acwt}.tra \
      2>$dir/rescore_mbr_${inv_acwt}.log
     
   cat $dir/${inv_acwt}.tra | \
    scripts/int2sym.pl --ignore-first-field $symtab | sed 's:<UNK>::g' | \
    compute-wer --text --mode=present ark:$dir/test_trans.filt  ark,p:-   >& $dir/wer_$inv_acwt
done

