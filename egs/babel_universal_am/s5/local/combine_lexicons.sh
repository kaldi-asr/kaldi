#!/bin/bash
# Copyright 2016  Johns Hopkins University (Author: Matthew Wiesner)
# Apache 2.0

list=
silence_lex=

. ./utils/parse_options.sh

if [ $# -le 1 ]; then
  echo >&2 "Usage: ./local/combine_lexicons.sh [--opts] <odict> dict1 dict2 dict3 ..."
  echo >&2 "  --list <path/to/list>"
  echo >&2 "  --silence-lexicon <path/to/silence/lexicon>: If empty, we assume the"
  echo >&2 "                                               BABEL silence lexicon."
  exit 1
fi

odict=$1
shift;

dicts=()
if [ ! -z $list ]; then
  echo "Detected list input"
  while read line; do
    dicts+=("$line")
  done < $list 
else
  for dict in $*; do
    dicts+=("$dict")
  done
fi

mkdir -p $odict
# Check that we have everything
for dict in ${dicts[@]}; do
  echo $dict
  for f in lexicon.txt extra_questions.txt silence_phones.txt nonsilence_phones.txt optional_silence.txt; do
    if [ ! -f ${dict}/${f} ]; then
      echo "Expected ${dict}/${f} to exist" && exit 1
    fi
  done
done

for dict in ${dicts[@]}; do
  cat ${dict}/lexicon.txt
done | sort -u > ${odict}/lexicon.txt

echo -e "<silence> SIL\n<unk>\t<oov>\n<noise>\t<sss>\n<v-noise>\t<vns>" > ${odict}/silence_lexicon.txt
./local/prepare_dict.py --silence-lexicon ${odict}/silence_lexicon.txt \
  $odict/lexicon.txt $odict
exit 0


