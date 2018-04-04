#!/bin/bash
# Copyright 2012  Johns Hopkins University (Author: Daniel Povey)
#           2014  Mff UK, UFAL (modification: Ondrej Platek)
# Apache 2.0

[ -f ./path.sh ] && . ./path.sh

# begin configuration section.
cmd=run.pl
min_lmw=9
max_lmw=20
#end configuration section.

[ -f ./path.sh ] && . ./path.sh
. parse_options.sh || exit 1;

if [ $# -ne 3 ]; then
  echo "Usage: local/score.sh [--cmd (run.pl|queue.pl...)] <data-dir> <lang-dir|graph-dir> <decode-dir>"
  echo " Options:"
  echo "    --cmd (run.pl|queue.pl...)      # specify how to run the sub-processes."
  echo "    --min_lmw <int>                # minumum LM-weight for lattice rescoring "
  echo "    --max_lmw <int>                # maximum LM-weight for lattice rescoring "
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

cp $data/text $dir/scoring/test.txt

$cmd LMW=$min_lmw:$max_lmw $dir/scoring/log/best_path.LMW.log \
  lattice-best-path --lm-scale=LMW --word-symbol-table=$symtab \
    "ark:gunzip -c $dir/lat.*.gz|" ark,t:$dir/scoring/LMW.tra || exit 1;

$cmd LMW=$min_lmw:$max_lmw $dir/scoring/log/score.LMW.log \
   cat $dir/scoring/LMW.tra \| \
    utils/int2sym.pl -f 2- $symtab \| \
    compute-wer --text --mode=present \
     ark:$dir/scoring/test.txt  ark,p:- ">&" $dir/wer_LMW || exit 1;

# Show results
for f in $dir/wer_*; do echo $f; egrep  '(WER)|(SER)' < $f; done

exit 0;
