#!/bin/bash
# Copyright 2012  Johns Hopkins University (Author: Daniel Povey)
# Apache 2.0

[ -f ./path.sh ] && . ./path.sh

cmd=run.pl
[ $1 == "--cmd" ] && cmd=$2 && shift 2;

[ $# -ne 3 ] && \
  echo "Usage: local/score.sh [--cmd (run.pl|queue.pl...)] <data-dir> <lang-dir|graph-dir> <decode-dir>" && exit 1;

data=$1
lang_or_graph=$2
dir=$3

symtab=$lang_or_graph/words.txt

for f in $symtab $dir/lat.1.gz $data/text; do
  [ ! -f $f ] && echo "score.sh: no such file $f" && exit 1;
done

mkdir -p $dir/scoring/log

cat $data/text | sed 's:<NOISE>::g' | sed 's:<SPOKEN_NOISE>::g' > $dir/scoring/test_filt.txt

$cmd LMWT=5:20 $dir/scoring/log/best_path.LMWT.log \
  lattice-best-path --lm-scale=LMWT --word-symbol-table=$symtab \
    "ark:gunzip -c $dir/lat.*.gz|" ark,t:$dir/scoring/LMWT.tra || exit 1;

# Note: the double level of quoting for the sed command
$cmd LMWT=5:20 $dir/scoring/log/score.LMWT.log \
   cat $dir/scoring/LMWT.tra \| \
    utils/int2sym.pl -f 2- $symtab \| sed 's:\<UNK\>::g' \| \
    compute-wer --text --mode=present \
     ark:$dir/scoring/test_filt.txt  ark,p:- ">&" $dir/wer_LMWT || exit 1;

exit 0;
