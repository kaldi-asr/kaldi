#!/bin/bash

# Script for minimum bayes risk decoding.

[ -f ./path.sh ] && . ./path.sh;

cmd=run.pl
[ $1 == "--cmd" ] && cmd=$2 && shift 2;

[ $# -ne 3 ] && \
  echo "Usage: local/score_mbr.sh [--cmd (run.pl|queue.pl...)] <data-dir> <lang-dir|graph-dir> <decode-dir>" && exit 1;


if [ $# -ne 3 ]; then
   echo "Usage: scripts/score_mbr.sh <decode-dir> <word-symbol-table> <data-dir>"
   exit 1;
fi

data=$1
lang_or_graph=$2
dir=$3

symtab=$lang_or_graph/words.txt

for f in $symtab $dir/lat.1.gz $data/text; do
  [ ! -f $f ] && echo "score_mbr.sh: no such file $f" && exit 1;
done

mkdir -p $dir/scoring/log

cat $data/text | sed 's:<NOISE>::g' | sed 's:<SPOKEN_NOISE>::g' > $dir/scoring/test_filt.txt

# We submit the jobs separately, not as an array, because it's hard
# to get the inverse of the LM scales.
rm $dir/.error 2>/dev/null
for inv_acwt in `seq 9 20`; do
  acwt=`perl -e "print (1.0/$inv_acwt);"`
  $cmd $dir/scoring/rescore_mbr.${inv_acwt}.log \
    lattice-mbr-decode  --acoustic-scale=$acwt --word-symbol-table=$symtab \
      "ark:gunzip -c $dir/lat.*.gz|" ark,t:$dir/scoring/${inv_acwt}.tra \
    || touch $dir/.error &
done
wait;
[ -f $dir/.error ] && echo "score_mbr.sh: errror getting MBR outout.";
     

$cmd LMWT=9:20 $dir/scoring/log/score.LMWT.log \
   cat $dir/scoring/LMWT.tra \| \
    utils/int2sym.pl -f 2- $symtab \| sed 's:\<UNK\>::g' \| \
    compute-wer --text --mode=present \
     ark:$dir/scoring/test_filt.txt  ark,p:- ">" $dir/wer_LMWT || exit 1;

