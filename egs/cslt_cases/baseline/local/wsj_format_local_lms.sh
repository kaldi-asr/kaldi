#!/bin/bash

# Copyright Johns Hopkins University (Author: Daniel Povey) 2012
#           Guoguo Chen 2014

lang_suffix=

echo "$0 $@"  # Print the command line for logging
. ./path.sh
. utils/parse_options.sh || exit 1;

[ ! -d data/lang${lang_suffix}_bd ] &&\
  echo "Expect data/local/lang${lang_suffix}_bd to exist" && exit 1;

lm_srcdir_3g=data/local/local_lm/3gram-mincount
lm_srcdir_4g=data/local/local_lm/4gram-mincount

[ ! -d "$lm_srcdir_3g" ] && echo "No such dir $lm_srcdir_3g" && exit 1;
[ ! -d "$lm_srcdir_4g" ] && echo "No such dir $lm_srcdir_4g" && exit 1;

for d in data/lang${lang_suffix}_test_bd_{tg,tgpr,tgconst,fg,fgpr,fgconst}; do
  rm -r $d 2>/dev/null
  cp -r data/lang${lang_suffix}_bd $d
done

lang=data/lang${lang_suffix}_bd

# Check a few files that we have to use.
for f in words.txt oov.int; do
  if [[ ! -f $lang/$f ]]; then
    echo "$0: no such file $lang/$f"
    exit 1;
  fi
done

# Parameters needed for ConstArpaLm.
unk=`cat $lang/oov.int`
bos=`grep "<s>" $lang/words.txt | awk '{print $2}'`
eos=`grep "</s>" $lang/words.txt | awk '{print $2}'`
if [[ -z $bos || -z $eos ]]; then
  echo "$0: <s> and </s> symbols are not in $lang/words.txt"
  exit 1;
fi

# Be careful: this time we dispense with the grep -v '<s> <s>' so this might
# not work for LMs generated from all toolkits.
gunzip -c $lm_srcdir_3g/lm_pr6.0.gz | \
  arpa2fst --disambig-symbol=#0 \
           --read-symbol-table=$lang/words.txt - data/lang${lang_suffix}_test_bd_tgpr/G.fst || exit 1;
  fstisstochastic data/lang${lang_suffix}_test_bd_tgpr/G.fst

gunzip -c $lm_srcdir_3g/lm_unpruned.gz | \
  arpa2fst --disambig-symbol=#0 \
           --read-symbol-table=$lang/words.txt - data/lang${lang_suffix}_test_bd_tg/G.fst || exit 1;
  fstisstochastic data/lang${lang_suffix}_test_bd_tg/G.fst

# Build ConstArpaLm for the unpruned language model.
gunzip -c $lm_srcdir_3g/lm_unpruned.gz | \
  utils/map_arpa_lm.pl $lang/words.txt | \
  arpa-to-const-arpa --bos-symbol=$bos --eos-symbol=$eos \
  --unk-symbol=$unk - data/lang${lang_suffix}_test_bd_tgconst/G.carpa || exit 1

gunzip -c $lm_srcdir_4g/lm_unpruned.gz | \
  arpa2fst --disambig-symbol=#0 \
           --read-symbol-table=$lang/words.txt - data/lang${lang_suffix}_test_bd_fg/G.fst || exit 1;
  fstisstochastic data/lang${lang_suffix}_test_bd_fg/G.fst

# Build ConstArpaLm for the unpruned language model.
gunzip -c $lm_srcdir_4g/lm_unpruned.gz | \
  utils/map_arpa_lm.pl $lang/words.txt | \
  arpa-to-const-arpa --bos-symbol=$bos --eos-symbol=$eos \
  --unk-symbol=$unk - data/lang${lang_suffix}_test_bd_fgconst/G.carpa || exit 1

gunzip -c $lm_srcdir_4g/lm_pr7.0.gz | \
  arpa2fst --disambig-symbol=#0 \
           --read-symbol-table=$lang/words.txt - data/lang${lang_suffix}_test_bd_fgpr/G.fst || exit 1;
  fstisstochastic data/lang${lang_suffix}_test_bd_fgpr/G.fst

exit 0;
