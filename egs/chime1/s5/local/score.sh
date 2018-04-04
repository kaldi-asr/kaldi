#!/bin/bash
# Copyright 2012  Johns Hopkins University (Author: Daniel Povey)
# Apache 2.0

# Copyright 2015  University of Sheffield (Author: Ning Ma)
# Apache 2.0.
#
# This script computes keyword (letter+digit) recognition accuracies across 
# SNRs for the 1st CHiME challenge

cmd=run.pl

[ -f ./path.sh ] && . ./path.sh
. parse_options.sh || exit 1;

if [ $# -ne 3 ]; then
  echo "Usage: local/score.sh [--cmd (run.pl|queue.pl...)] <data-dir> <lang-dir|graph-dir> <decode-dir>"
  echo " Options:"
  echo "    --cmd (run.pl|queue.pl...)      # specify how to run the sub-processes."
  exit 1;
fi

data=$1
lang_or_graph=$2
dir=$3

symtab=$lang_or_graph/words.txt

for f in $symtab $dir/lat.1.gz $data/text; do
  [ ! -f $f ] && echo "score.sh: no such file $f" && exit 1;
done

mkdir -p $dir/scoring/log

$cmd $dir/scoring/log/best_path.log \
  lattice-best-path --word-symbol-table=$symtab \
    "ark:gunzip -c $dir/lat.*.gz|" ark,t:$dir/scoring/trans.int || exit 1;

cat $dir/scoring/trans.int | utils/int2sym.pl -f 2- $symtab \
  | sed 's:\<UNK\>::g;s:<SIL>::g' > $dir/scoring/trans.txt

local/compute_chime1_scores.pl $dir/scoring/trans.txt > $dir/keyword_scores.txt

echo "Scores are available in $dir/keyword_scores.txt"
cat $dir/keyword_scores.txt
exit 0;

