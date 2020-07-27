#!/usr/bin/env bash
# Copyright 2012  Johns Hopkins University (Author: Daniel Povey)
# Apache 2.0

[ -f ./path.sh ] && . ./path.sh

# begin configuration section.
cmd=run.pl
stage=0
decode_mbr=true
reverse=false
word_ins_penalty=0.0
min_lmwt=9
max_lmwt=30

#end configuration section.

[ -f ./path.sh ] && . ./path.sh
. parse_options.sh || exit 1;

if [ $# -ne 3 ]; then
  echo "Usage: local/score.sh [--cmd (run.pl|queue.pl...)] <data-dir> <lang-dir|graph-dir> <decode-dir>"
  echo " Options:"
  echo "    --cmd (run.pl|queue.pl...)      # specify how to run the sub-processes."
  echo "    --stage (0|1|2)                 # start scoring script from part-way through."
  echo "    --decode_mbr (true/false)       # maximum bayes risk decoding (confusion network)."
  echo "    --min_lmwt <int>                # minumum LM-weight for lattice rescoring "
  echo "    --max_lmwt <int>                # maximum LM-weight for lattice rescoring "
  echo "    --reverse (true/false)          # score with time reversed features "
  exit 1;
fi

args=$*

data=$1
lang_or_graph=$2
dir=$3
srcdir=`dirname $dir`;
symtab=$lang_or_graph/words.txt

for f in $symtab $dir/lat.1.gz; do 
  [ ! -f $f ] && echo "score.sh: no such file $f" && exit 1;
done

if [ -f $data/stm ]; then
  local/score_sclite.sh --decode-mbr $decode_mbr --reverse $reverse --word-ins-penalty $word_ins_penalty --min-lmwt $min_lmwt --max-lmwt $max_lmwt --stage $stage --cmd "$cmd" $args
else
  steps/score_kaldi.sh --decode-mbr $decode_mbr --word-ins-penalty $word_ins_penalty --min-lmwt $min_lmwt --max-lmwt $max_lmwt --stage $stage --cmd "$cmd" $args
fi
