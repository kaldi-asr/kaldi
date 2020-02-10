#!/usr/bin/env bash

# Copyright 2019 Johns Hopkins University (author: Jinyi Yang)
# Apache 2.0

# This script merges the gale-tdt lexicon dicrectory with gigaword (simplified Mandarin)
# lexicon directory. It requires the lexiconp.txt file in both directories
# since the probabilities in lexiconp.txt may be re-estimated.

if [ $# -ne 3 ];then
  echo "Usage: $0 <gale-tdt-dict-dir> <giga-dict-dir> <tgt-lex-dir>"
  echo "E.g., $0 data/local/dict_gale_tdt data/local/dict_giga data/local/dict_merged"
  exit 1
fi

lex_dir_1=$1
lex_dir_2=$2
tgt_lex_dir=$3

mkdir -p $tgt_lex_dir

for f in silence_phones.txt nonsilence_phones.txt lexiconp.txt extra_questions.txt;do
  [ ! -f $lex_dir_1/$f ] && echo "$0: no such file $lex_dir_1/$f" && exit 1;
  [ ! -f $lex_dir_2/$f ] && echo "$0: no such file $lex_dir_2/$f" && exit 1;
  # We copy the phone related files from gale dictionary directory, since they
  # are the same phone sets as GIGA words.
  cp $lex_dir_1/$f $tgt_lex_dir
done

mv $tgt_lex_dir/lexiconp.txt $tgt_lex_dir/lexiconp_1.txt


awk 'NR==FNR{a[$1];next}{if (!($1 in a)) print $0}' $tgt_lex_dir/lexiconp_1.txt \
  $lex_dir_2/lexiconp.txt > $tgt_lex_dir/lexiconp_2.txt
cat $tgt_lex_dir/lexiconp_1.txt $tgt_lex_dir/lexiconp_2.txt | sort > $tgt_lex_dir/lexiconp.txt


