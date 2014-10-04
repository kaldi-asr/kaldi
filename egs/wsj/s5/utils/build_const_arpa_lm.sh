#!/bin/bash

# Copyright 2014  Guoguo Chen
# Apache 2.0

# This script reads in an Arpa format language model, and converts it into the
# ConstArpaLm format language model.

# begin configuration section
# end configuration section

[ -f path.sh ] && . ./path.sh;

. utils/parse_options.sh

if [ $# != 3 ]; then
  echo "Usage: "
  echo "  $0 [options] <arpa-lm-path> <old-lang-dir> <new-lang-dir>"
  echo "e.g.:"
  echo "  $0 data/local/lm/3-gram.full.arpa.gz data/lang/ data/lang_test_tgmed"
  echo "Options"
  exit 1;
fi

export LC_ALL=C

arpa_lm=$1
old_lang=$2
new_lang=$3

mkdir -p $new_lang

for f in L_disambig.fst \
  L.fst oov.int oov.txt phones.txt topo words.txt phones; do
  if [[ ! -f $old_lang/$f && ! -d $old_lang/$f ]]; then
    echo "$0: no such file or directory $old_lang/$f"
    exit 1;
  fi
  cp -rf $old_lang/$f $new_lang
done

# First, convert the words in the Arpa format language model into integers.
arpa_lm_int=$new_lang/arpa.int


# Second, convert $arpa_lm_int to ConstArpaLm format.
unk=`cat $new_lang/oov.int`
bos=`grep "<s>" $new_lang/words.txt | awk '{print $2}'`
eos=`grep "</s>" $new_lang/words.txt | awk '{print $2}'`
if [[ -z $bos || -z $eos ]]; then
  echo "$0: <s> and </s> symbols are not in $new_lang/words.txt"
  exit 1
fi

gunzip -c $arpa_lm | utils/map_arpa_lm.pl $new_lang/words.txt | \
  arpa-to-const-arpa --bos-symbol=$bos \
   --eos-symbol=$eos --unk-symbol=$unk - $new_lang/G.carpa || exit 1

