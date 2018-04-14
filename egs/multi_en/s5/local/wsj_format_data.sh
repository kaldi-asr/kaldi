#!/bin/bash

###########################################################################################
# This script was copied from egs/wsj/s5/local/wsj_format_data.sh
# The source commit was e69198c3dc5633f98eb88e1cdf20b2521a598f21
# Changes made:
#  - Modified paths to match multi_en naming conventions
#  - Removed code related to LM creation
#  - Only prepared data/wsj/train_si284 (no test sets)
###########################################################################################

# Copyright 2012  Microsoft Corporation  Johns Hopkins University (Author: Daniel Povey)
#           2015  Guoguo Chen
# Apache 2.0

# This script takes data prepared in a corpus-dependent way
# in data/local/, and converts it into the "canonical" form,
# in various subdirectories of data/, e.g. data/lang, data/lang_test_ug,
# data/train_si284, data/train_si84, etc.

# Don't bother doing train_si84 separately (although we have the file lists
# in data/local/) because it's just the first 7138 utterances in train_si284.
# We'll create train_si84 after doing the feature extraction.

lang_suffix=

echo "$0 $@"  # Print the command line for logging
. utils/parse_options.sh || exit 1;

. ./path.sh || exit 1;

echo "Preparing train and test data"
srcdir=data/local/wsj/data
lmdir=data/local/wsj/nist_lm
tmpdir=data/local/wsj/lm_tmp
lexicon=data/local/wsj/lang${lang_suffix}_tmp/lexiconp.txt
mkdir -p $tmpdir

#for x in train_si284 test_eval92 test_eval93 test_dev93 test_eval92_5k test_eval93_5k test_dev93_5k dev_dt_05 dev_dt_20; do
for x in train_si284; do
  mkdir -p data/wsj/$x
  cp $srcdir/${x}_wav.scp data/wsj/$x/wav.scp || exit 1;
  cp $srcdir/$x.txt data/wsj/$x/text || exit 1;
  cp $srcdir/$x.spk2utt data/wsj/$x/spk2utt || exit 1;
  cp $srcdir/$x.utt2spk data/wsj/$x/utt2spk || exit 1;
  utils/filter_scp.pl data/wsj/$x/spk2utt $srcdir/spk2gender > data/wsj/$x/spk2gender || exit 1;
done


# Removed LM-related stuff


echo "Succeeded in formatting data."
rm -r $tmpdir
