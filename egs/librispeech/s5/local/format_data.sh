#!/bin/bash

# Copyright 2014 Vassil Panayotov
# Apache 2.0

# Prepares the test time language model(G) transducers
# (adapted from wsj/s5/local/wsj_format_data.sh)

. path.sh

if [ $# -ne 1 ]; then
  echo "Usage: $0 <lm-dir>"
  echo "e.g.: $0 /export/a15/vpanayotov/data/lm"
  echo ", where:"
  echo "    <lm-dir> is the directory in which the language model is stored/downloaded"
  exit 1
fi

lm_dir=$1

tmpdir=data/local/lm_tmp
lexicon=data/local/lang_tmp/lexiconp.txt
mkdir -p $tmpdir

# This loop was taken verbatim from wsj_format_data.sh, and I'm leaving it in place in
# case we decide to add more language models at some point
for lm_suffix in tgpr; do
  test=data/lang_test_${lm_suffix}
  mkdir -p $test
  for f in phones.txt words.txt phones.txt L.fst L_disambig.fst phones oov.txt oov.int; do
    cp -r data/lang/$f $test
  done
  gunzip -c $lm_dir/lm_${lm_suffix}.arpa.gz |\
   utils/find_arpa_oovs.pl $test/words.txt  > $tmpdir/oovs_${lm_suffix}.txt || exit 1

  # grep -v '<s> <s>' because the LM seems to have some strange and useless
  # stuff in it with multiple <s>'s in the history.  Encountered some other similar
  # things in a LM from Geoff.  Removing all "illegal" combinations of <s> and </s>,
  # which are supposed to occur only at being/end of utt.  These can cause
  # determinization failures of CLG [ends up being epsilon cycles].
  gunzip -c $lm_dir/lm_${lm_suffix}.arpa.gz | \
    grep -v '<s> <s>' | \
    grep -v '</s> <s>' | \
    grep -v '</s> </s>' | \
    arpa2fst - | fstprint | \
    utils/remove_oovs.pl $tmpdir/oovs_${lm_suffix}.txt | \
    utils/eps2disambig.pl | utils/s2eps.pl | fstcompile --isymbols=$test/words.txt \
      --osymbols=$test/words.txt  --keep_isymbols=false --keep_osymbols=false | \
     fstrmepsilon | fstarcsort --sort_type=ilabel > $test/G.fst

  utils/validate_lang.pl $test || exit 1;
done

echo "Succeeded in formatting data."
rm -r $tmpdir

exit 0
