#!/bin/bash

# Copyright Johns Hopkins University (Author: Daniel Povey) 2012

. ./path.sh

[ ! -d data/lang_bd ] && echo "Expect data/local/lang_bd to exist" && exit 1;

lm_srcdir_3g=data/local/local_lm/3gram-mincount
lm_srcdir_4g=data/local/local_lm/4gram-mincount

[ ! -d "$lm_srcdir_3g" ] && echo "No such dir $lm_srcdir_3g" && exit 1;
[ ! -d "$lm_srcdir_4g" ] && echo "No such dir $lm_srcdir_4g" && exit 1;

for d in data/lang_test_bd_{tg,tgpr,fg,fgpr}; do
  rm -r $d 2>/dev/null
  cp -r data/lang_bd $d
done

lang=data/lang_bd

# Be careful: this time we dispense with the grep -v '<s> <s>' so this might
# not work for LMs generated from all toolkits.
gunzip -c $lm_srcdir_3g/lm_pr6.0.gz | \
  arpa2fst - | fstprint | \
    utils/eps2disambig.pl | utils/s2eps.pl | fstcompile --isymbols=$lang/words.txt \
      --osymbols=$lang/words.txt  --keep_isymbols=false --keep_osymbols=false | \
     fstrmepsilon > data/lang_test_bd_tgpr/G.fst || exit 1;
  fstisstochastic data/lang_test_bd_tgpr/G.fst

gunzip -c $lm_srcdir_3g/lm_unpruned.gz | \
  arpa2fst - | fstprint | \
    utils/eps2disambig.pl | utils/s2eps.pl | fstcompile --isymbols=$lang/words.txt \
      --osymbols=$lang/words.txt  --keep_isymbols=false --keep_osymbols=false | \
     fstrmepsilon > data/lang_test_bd_tg/G.fst || exit 1;
  fstisstochastic data/lang_test_bd_tg/G.fst

gunzip -c $lm_srcdir_4g/lm_unpruned.gz | \
  arpa2fst - | fstprint | \
    utils/eps2disambig.pl | utils/s2eps.pl | fstcompile --isymbols=$lang/words.txt \
      --osymbols=$lang/words.txt  --keep_isymbols=false --keep_osymbols=false | \
     fstrmepsilon > data/lang_test_bd_fg/G.fst || exit 1;
  fstisstochastic data/lang_test_bd_fg/G.fst

gunzip -c $lm_srcdir_4g/lm_pr7.0.gz | \
  arpa2fst - | fstprint | \
    utils/eps2disambig.pl | utils/s2eps.pl | fstcompile --isymbols=$lang/words.txt \
      --osymbols=$lang/words.txt  --keep_isymbols=false --keep_osymbols=false | \
     fstrmepsilon > data/lang_test_bd_fgpr/G.fst || exit 1;
  fstisstochastic data/lang_test_bd_fgpr/G.fst

exit 0;
