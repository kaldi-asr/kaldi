#!/usr/bin/env bash

# Copyright 2014 QCRI (author: Ahmed Ali)
# Apache 2.0

if [ -f path.sh ]; then
  . ./path.sh; else
   echo "missing path.sh"; exit 1;
fi

if [ $# -ne 3 ]; then
  echo "Usage: $0 <lm-file> <src-dir> <tgt-dir>"
  echo "E.g., $0 data/local/lm/srim.o4g.kn.gz data/lang data/lang_test"
  exit 1
fi

arpa_lm=$1
src_dir=$2
tgt_dir=$3


set -e -o pipefail
set -x

export LC_ALL=C

#arpa_lm=data/local/gale/train/lm_4gram/srilm.o4g.kn.gz

[ ! -f $arpa_lm ] && echo No such file $arpa_lm && exit 1;

rm -r $tgt_dir || true
cp -r $src_dir $tgt_dir

gunzip -c "$arpa_lm" | \
  arpa2fst --disambig-symbol=#0 \
           --read-symbol-table=$tgt_dir/words.txt - $tgt_dir/G.fst


echo  "Checking how stochastic G is (the first of these numbers should be small):"
fstisstochastic $tgt_dir/G.fst || true

## Check lexicon.
## just have a look and make sure it seems sane.
echo "First few lines of lexicon FST:"
(
  fstprint   --isymbols=$src_dir/phones.txt --osymbols=$src_dir/words.txt $src_dir/L.fst | head
) || true
echo Performing further checks

# Checking that G.fst is determinizable.
fstdeterminize $tgt_dir/G.fst /dev/null || {
  echo Error determinizing G.
  exit 1
}

# Checking that L_disambig.fst is determinizable.
fstdeterminize $tgt_dir/L_disambig.fst /dev/null || echo Error determinizing L.

# Checking that disambiguated lexicon times G is determinizable
# Note: we do this with fstdeterminizestar not fstdeterminize, as
# fstdeterminize was taking forever (presumbaly relates to a bug
# in this version of OpenFst that makes determinization slow for
# some case).
fsttablecompose $tgt_dir/L_disambig.fst $tgt_dir/G.fst | \
   fstdeterminizestar >/dev/null || echo Error

# Checking that LG is stochastic:
fsttablecompose $tgt_dir/L_disambig.fst $tgt_dir/G.fst | \
   fstisstochastic || echo LG is not stochastic

echo "LM preparation succeeded."
