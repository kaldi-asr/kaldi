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
lmdir=data/local/arpa_lm
tmpdir=data/local/lm_tmp
lang_tmp=data/local/lang_tmp
lexicon=data/local/dict/transcripts
ccs=data/local/lang_tmp/cmuclmtk.ccs
lm_suffix=arpa
mkdir -p $lmdir
mkdir -p $tmpdir

# Create context cue symbol file for cmuclmtk
echo -e '<s>' > $ccs
echo -e '</s>' >> $ccs


# Envelop LM training data in context cues
python3 local/sprak_prep_lm.py $lexicon $lmdir/lm_input


# Next, for each type of language model, create the corresponding FST
# and the corresponding lang_test_* directory.

echo Preparing language models for test

text2wfreq < $lmdir/lm_input | wfreq2vocab -top 40000 > $lmdir/sprak.vocab

text2idngram -vocab $lmdir/sprak.vocab -idngram $lmdir/sprak.idngram < $lmdir/lm_input

idngram2lm -linear -idngram $lmdir/sprak.idngram -vocab \
    $lmdir/sprak.vocab -arpa $lmdir/sprak.arpa -context $ccs


test=data/lang_test_${lm_suffix}
mkdir -p $test
cp -r data/lang/* $test

cat $lmdir/sprak.arpa | \
  arpa2fst --disambig-symbol=#0 \
           --read-symbol-table=$test/words.txt - $test/G.fst


utils/validate_lang.pl $test || exit 1;

exit 0;
