#!/usr/bin/env bash

# Copyright 2014 QCRI (author: Ahmed Ali)
# Apache 2.0

if [ -f path.sh ]; then
  . ./path.sh; else
   echo "missing path.sh"; exit 1;
fi

for dir in test train; do
   cp -pr data/local/$dir data/$dir
done


mkdir -p data/lang_test

arpa_lm=data/local/lm/3gram-mincount/lm_unpruned.gz
[ ! -f $arpa_lm ] && echo No such file $arpa_lm && exit 1;

rm -r data/lang_test
cp -r data/lang data/lang_test

gunzip -c "$arpa_lm" | \
  arpa2fst --disambig-symbol=#0 \
           --read-symbol-table=data/lang_test/words.txt - data/lang_test/G.fst


echo  "Checking how stochastic G is (the first of these numbers should be small):"
fstisstochastic data/lang_test/G.fst

## Check lexicon.
## just have a look and make sure it seems sane.
echo "First few lines of lexicon FST:"
fstprint   --isymbols=data/lang/phones.txt --osymbols=data/lang/words.txt data/lang/L.fst  | head

echo Performing further checks

# Checking that G.fst is determinizable.
fstdeterminize data/lang_test/G.fst /dev/null || echo Error determinizing G.

# Checking that L_disambig.fst is determinizable.
fstdeterminize data/lang_test/L_disambig.fst /dev/null || echo Error determinizing L.

# Checking that disambiguated lexicon times G is determinizable
# Note: we do this with fstdeterminizestar not fstdeterminize, as
# fstdeterminize was taking forever (presumbaly relates to a bug
# in this version of OpenFst that makes determinization slow for
# some case).
fsttablecompose data/lang_test/L_disambig.fst data/lang_test/G.fst | \
   fstdeterminizestar >/dev/null || echo Error

# Checking that LG is stochastic:
fsttablecompose data/lang/L_disambig.fst data/lang_test/G.fst | \
   fstisstochastic || echo LG is not stochastic


echo gale_format_data  succeeded.

exit 0
