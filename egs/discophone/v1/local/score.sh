#!/usr/bin/env bash
# Copyright 2012  Johns Hopkins University (Author: Daniel Povey)
#           2014  Guoguo Chen
#           2020  Johns Hopkins University (Author: Piotr Żelasko)
# Apache 2.0

[ -f ./path.sh ] && . ./path.sh

# begin configuration section.
cmd=run.pl
word_ins_penalty=0.0,0.5,1.0
min_lmwt=7
max_lmwt=17
#end configuration section.

[ -f ./path.sh ] && . ./path.sh
. parse_options.sh || exit 1

if [ $# -ne 3 ]; then
  echo "Usage: local/score.sh [--cmd (run.pl|queue.pl...)] <data-dir> <lang-dir|graph-dir> <decode-dir>"
  echo " Options:"
  echo "    --cmd (run.pl|queue.pl...)      # specify how to run the sub-processes."
  echo "    --min_lmwt <int>                # minumum LM-weight for lattice rescoring "
  echo "    --max_lmwt <int>                # maximum LM-weight for lattice rescoring "
  exit 1
fi

data=$1
lang_or_graph=$2
dir=$3

symtab=$lang_or_graph/words.txt

for f in $symtab $dir/lat.1.gz $data/text; do
  [ ! -f $f ] && echo "score.sh: no such file $f" && exit 1
done

mkdir -p $dir/scoring/log

cat $data/text | sed 's:<noise>::g' >$dir/scoring/test_filt.txt

for wip in $(echo $word_ins_penalty | sed 's/,/ /g'); do
  $cmd LMWT=$min_lmwt:$max_lmwt $dir/scoring/log/best_path.LMWT.$wip.log \
    lattice-scale --inv-acoustic-scale=LMWT "ark:gunzip -c $dir/lat.*.gz|" ark:- \| \
    lattice-add-penalty --word-ins-penalty=$wip ark:- ark:- \| \
    lattice-best-path --word-symbol-table=$symtab \
    ark:- ark,t:$dir/scoring/LMWT.$wip.tra || exit 1
done

# Note: the double level of quoting for the sed command
for wip in $(echo $word_ins_penalty | sed 's/,/ /g'); do
  $cmd LMWT=$min_lmwt:$max_lmwt $dir/scoring/log/score.LMWT.$wip.log \
    cat $dir/scoring/LMWT.$wip.tra \| \
    utils/int2sym.pl -f 2- $symtab \| sed 's:\<unk\>::g' \| \
    compute-wer --text --mode=present \
    ark:$dir/scoring/test_filt.txt ark,p:- ">&" $dir/wer_LMWT_$wip || exit 1
done

grep WER $dir/wer* | utils/best_wer.sh | sed 's/WER/PER/g'
