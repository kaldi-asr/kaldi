#!/bin/bash
# Copyright 2016  Johns Hopkins University (Author: Matthew Wiesner)
# Apache 2.0

# This script presupposes a manual mapping from the missing phones to existsing
# phones already exists.


if [ $# -lt 6 ]; then
  echo >&2 "Usage: ./local/convert_dict.sh <odict> <olang> <idict1> <idict2> <ilang> <map>"
  echo >&2 ""
  echo >&2 "Description -- This script takes as input a kaldi style dictionary "
  echo >&2 "and makes modifications to the phonemes in the lexicon, "
  echo >&2 "<idict1>/lexicon.txt, that do no occur in <idict2>/lexion.txt, "
  echo >&2 "according to <map>. Silence phonemes / words in both lexicons are "
  echo >&2 "merged if present, but otherwise, the default BABEL silence words"
  echo >&2 "and phonemes are used. The resulting dictionary directory is"
  echo >&2 "created and is used to build a the kaldi lang directory. The phone"
  echo >&2 "symbols are shared using the --phone-symbol-table option of"
  echo >&2 "utils/prepare_lang.sh"
  echo >&2 "-----------------"
  echo >&2 "<odict> -- The converted (output) kaldi-style dictionary directory"
  echo >&2 "<olang> -- The converted (output) kaldi-style lang directory"
  echo >&2 "<idict1> -- The input kaldi-style dictionary directory to convert"
  echo >&2 "<idict2> -- The kaldi style dictionary directory whose silence "
  echo >&2 "            phonemes should be included if present."
  echo >&2 "<ilang> -- The lang directory whose phone-symbol-table should be"
  echo >&2 "           used to ensure that the phonemes in <olang> get mapped"
  echo >&2 "           to the same integers (and hence acoustic models later)"
  echo >&2 "<map>   -- A map from the phones in <idict1>/lexicon.txt "
  echo >&2 "           missing from <idict2>/lexicon.txt to existing phones in"
  echo >&2 "           <idict2>/lexicon.txt"
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

# NOTE. We are using the --phone-symbol-table option
./utils/prepare_lang.sh --share-silence-phones true \
                        --phone-symbol-table ${ilang}/phones.txt \
                        $odict "<unk>" ${odict}/tmp.lang $olang

exit 0
