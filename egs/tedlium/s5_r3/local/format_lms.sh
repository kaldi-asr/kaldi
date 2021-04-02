#!/usr/bin/env bash
#
# Copyright  2014 Nickolay V. Shmyrev
# Apache 2.0

if [ -f path.sh ]; then . path.sh; fi


small_arpa_lm=data/local/local_lm/data/arpa/4gram_small.arpa.gz
big_arpa_lm=data/local/local_lm/data/arpa/4gram_big.arpa.gz

for f in $small_arpa_lm $big_arpa_lm data/lang_nosp/words.txt; do
  [ ! -f $f ] && echo "$0: expected file $f to exist" && exit 1
done


set -e

if [ -f data/lang_nosp/G.fst ] && [ data/lang_nosp/G.fst -nt $small_arpa_lm ]; then
  echo "$0: not regenerating data/lang_nosp/G.fst as it already exists and "
  echo ".. is newer than the source LM."
else
  arpa2fst --disambig-symbol=#0 --read-symbol-table=data/lang_nosp/words.txt \
    "gunzip -c $small_arpa_lm|" data/lang_nosp/G.fst
  echo  "$0: Checking how stochastic G is (the first of these numbers should be small):"
  fstisstochastic data/lang_nosp/G.fst || true
  utils/validate_lang.pl --skip-determinization-check data/lang_nosp
fi



if [ -f data/lang_nosp_rescore/G.carpa ] && [ data/lang_nosp_rescore/G.carpa -nt $big_arpa_lm ] && \
    [ data/lang_nosp_rescore/G.carpa -nt data/lang_nosp/words.txt ]; then
  echo "$0: not regenerating data/lang_nosp_rescore/ as it seems to already by up to date."
else
  utils/build_const_arpa_lm.sh $big_arpa_lm data/lang_nosp data/lang_nosp_rescore || exit 1;
fi

exit 0;
