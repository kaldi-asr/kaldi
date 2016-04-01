#!/bin/bash 
#
# Copyright  2014 Nickolay V. Shmyrev 
# Apache 2.0

lang=data/lang_nosp

if [ -f path.sh ]; then . path.sh; fi

. utils/parse_options.sh

arpa_lm=db/cantab-TEDLIUM/cantab-TEDLIUM-pruned.lm3.gz 
[ ! -f $arpa_lm ] && echo No such file $arpa_lm && exit 1;

rm -rf ${lang}_test
cp -r ${lang} ${lang}_test

# grep -v '<s> <s>' etc. is only for future-proofing this script.  Our
# LM doesn't have these "invalid combinations".  These can cause 
# determinization failures of CLG [ends up being epsilon cycles].
# Note: remove_oovs.pl takes a list of words in the LM that aren't in
# our word list.  Since our LM doesn't have any, we just give it
# /dev/null [we leave it in the script to show how you'd do it].
gunzip -c "$arpa_lm" | \
   grep -v '<s> <s>' | \
   grep -v '</s> <s>' | \
   grep -v '</s> </s>' | \
   arpa2fst - | fstprint | \
   utils/remove_oovs.pl /dev/null | \
   utils/eps2disambig.pl | utils/s2eps.pl | fstcompile --isymbols=${lang}_test/words.txt \
     --osymbols=${lang}_test/words.txt  --keep_isymbols=false --keep_osymbols=false | \
    fstrmepsilon | fstarcsort --sort_type=ilabel > ${lang}_test/G.fst


echo  "Checking how stochastic G is (the first of these numbers should be small):"
fstisstochastic ${lang}_test/G.fst

utils/validate_lang.pl ${lang}_test || exit 1;

if [ ! -d ${lang}_rescore ]; then

  big_arpa_lm=db/cantab-TEDLIUM/cantab-TEDLIUM-unpruned.lm4.gz
  [ ! -f $big_arpa_lm ] && echo No such file $big_arpa_lm && exit 1;

  utils/build_const_arpa_lm.sh $big_arpa_lm ${lang}_test ${lang}_rescore || exit 1;

fi

exit 0;
