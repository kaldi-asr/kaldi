#!/bin/bash

# Copyright 2012  Microsoft Corporation  Johns Hopkins University (Author: Daniel Povey)
# Apache 2.0

# This script takes data prepared in a corpus-dependent way
# in data/local/, and converts it into the "canonical" form,
# in various subdirectories of data/, e.g. data/lang, data/lang_test_ug,
# data/train_si284, data/train_si84, etc.

# Don't bother doing train_si84 separately (although we have the file lists
# in data/local/) because it's just the first 7138 utterances in train_si284.
# We'll create train_si84 after doing the feature extraction.

. ./path.sh || exit 1;

echo "Preparing train and test data"
srcdir=data/local/data
lmdir=data/local/nist_lm
tmpdir=data/local/lm_tmp
lexicon=data/local/lang_tmp/lexiconp.txt
mkdir -p $tmpdir

for x in train_si84_clean train_si84_multi test_eval92 test_0166 dev_0330 dev_1206; do
  mkdir -p data/$x
  cp $srcdir/${x}_wav.scp data/$x/wav.scp || exit 1;
  cp $srcdir/$x.txt data/$x/text || exit 1;
  cp $srcdir/$x.spk2utt data/$x/spk2utt || exit 1;
  cp $srcdir/$x.utt2spk data/$x/utt2spk || exit 1;
  utils/filter_scp.pl data/$x/spk2utt $srcdir/spk2gender > data/$x/spk2gender || exit 1;
done


# Next, for each type of language model, create the corresponding FST
# and the corresponding lang_test_* directory.

echo Preparing language models for test

for lm_suffix in bg tgpr tg bg_5k tgpr_5k tg_5k; do
  test=data/lang_test_${lm_suffix}
  mkdir -p $test
  cp -r data/lang/* $test

  gunzip -c $lmdir/lm_${lm_suffix}.arpa.gz | \
    arpa2fst --disambig-symbol=#0 \
             --read-symbol-table=$test/words.txt - $test/G.fst

  utils/validate_lang.pl --skip-determinization-check $test || exit 1;
done

echo "Succeeded in formatting data."
rm -r $tmpdir
