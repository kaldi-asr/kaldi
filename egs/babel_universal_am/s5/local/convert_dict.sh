#!/bin/bash
# Copyright 2016  Johns Hopkins University (Author: Matthew Wiesner)
# Apache 2.0

# This script presupposes a manual mapping from the missing phones to existsing
# phones already exists.


if [ $# -lt 6 ]; then
  echo >&2 "Usage: ./local_/convert_lang.sh <odict> <olang> <idict1> <idict2> <ilang> <map>"
  exit 1
fi

odict=$1
olang=$2
idict1=$3
idict2=$4
ilang=$5
map=$6

mkdir -p $odict
if [ -f ${idict1}/silence_lexicon.txt ] && [ -f ${idict2}/silence_lexicon.txt ]; then
  cat ${idict1}/silence_lexicon.txt ${idict2}/silence_lexicon.txt \
    | sort -u > ${odict}/silence_lexicon.txt
else
  # Use BABEL default silence lexicon.
  echo -e "<silence> SIL\n<unk> <oov>\n<noise> <sss>\n<v-noise> <vns>" \
    > ${odict}/silence_lexicon.txt
fi

cat ${idict1}/lexicon.txt | utils/apply_map.pl -f 2- --permissive $map 2>/dev/null > ${odict}/lexicon.txt

./local/prepare_dict.py --silence-lexicon ${odict}/silence_lexicon.txt \
                                   ${odict}/lexicon.txt $odict

./utils/prepare_lang.sh --share-silence-phones true \
                        --phone-symbol-table ${ilang}/phones.txt \
                        $odict "<unk>" ${odict}/tmp.lang $olang

exit 0
