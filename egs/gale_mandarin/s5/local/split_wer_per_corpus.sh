#!/bin/bash

# Report WER for reports and conversational
# Copyright 2014 QCRI (author: Ahmed Ali)
# Apache 2.0

if [ $# -ne 1 ]; then
   echo "Arguments should be the gale folder, see ../run.sh for example."
   exit 1;
fi

[ -f ./path.sh ] && . ./path.sh

set -o pipefail -e

galeFolder=$(readlink -f $1)
symtab=./data/lang/words.txt

min_lmwt=7
max_lmwt=20

for dir in exp/*/*decode*; do
  for type in $(ls -1 local/test.* | xargs -n1 basename); do
    rm -fr $dir/scoring_$type
    mkdir -p $dir/scoring_$type/log
    for x in $dir/scoring/*.char $dir/scoring/*.tra $dir/scoring/char.filt $dir/scoring/text.filt; do
      cat $x | grep -f local/$type > $dir/scoring_$type/$(basename $x)
    done

    utils/run.pl LMWT=$min_lmwt:$max_lmwt $dir/scoring_$type/log/score.LMWT.log \
       cat $dir/scoring_${type}/LMWT.tra \| \
       utils/int2sym.pl -f 2- $symtab \| sed 's:\<UNK\>::g' \| \
       compute-wer --text --mode=present \
       ark:$dir/scoring_${type}/text.filt  ark,p:- ">&" $dir/wer_${type}_LMWT
    utils/run.pl LMWT=$min_lmwt:$max_lmwt $dir/scoring_$type/log/score.cer.LMWT.log \
       cat $dir/scoring_${type}/LMWT.char \| \
       compute-wer --text --mode=present \
       ark:$dir/scoring_${type}/char.filt  ark,p:- ">&" $dir/cer_${type}_LMWT
  done
done

for type in $(ls -1 local/test.* | xargs -n1 basename); do
  echo -e "\n# WER $type"
  for x in exp/*/*decode*; do
    grep WER $x/wer_${type}_* | utils/best_wer.sh;
  done | sort -n -k2
done

for type in $(ls -1 local/test.* | xargs -n1 basename); do
  echo -e "\n# CER $type"
  for x in exp/*/*decode*; do
    grep WER $x/cer_${type}_* | utils/best_wer.sh;
  done | sort -n -k2
done



