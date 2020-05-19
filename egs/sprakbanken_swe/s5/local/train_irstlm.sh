#!/usr/bin/env bash

# Copyright 2013  Mirsk Digital ApS (Author: Andreas Kirkedal)
# Apache 2.0

# This script takes data prepared in a corpus-dependent way
# in data/local/, and converts it into the "canonical" form,
# in various subdirectories of data/, e.g. data/lang, data/lang_test_ug,
# data/train_si284, data/train_si84, etc.

# Don't bother doing train_si84 separately (although we have the file lists
# in data/local/) because it's just the first 7138 utterances in train_si284.
# We'll create train_si84 after doing the feature extraction.

. ./path.sh || exit 1;

if [ -z $IRSTLM ] ; then
  export IRSTLM=$KALDI_ROOT/tools/irstlm/
fi
export PATH=${PATH}:$IRSTLM/bin
if ! command -v ngt >/dev/null 2>&1 ; then
  echo "$0: Error: the IRSTLM is not available or compiled" >&2
  echo "$0: Error: We used to install it by default, but." >&2
  echo "$0: Error: this is no longer the case." >&2
  echo "$0: Error: To install it, go to $KALDI_ROOT/tools" >&2
  echo "$0: Error: and run extras/install_irstlm.sh" >&2
  exit 1
fi

echo "Preparing train and test data"
srcdir=$4
lmdir=$5
tmpdir=data/local/lm_tmp
lang_tmp=data/local/lang_tmp
lexicon=$1
ngram=$2
lm_suffix=$3
mkdir -p $lmdir
mkdir -p $tmpdir


#grep -P -v '^[\s?|\.|\!]*$' $lexicon | grep -v '^ *$' | \
#awk '{if(NF>=4){ printf("%s\n",$0); }}' > $lmdir/text.filt

# Envelop LM training data in context cues
add-start-end.sh < $lexicon | awk '{if(NF>=3){ printf("%s\n",$0); }}' > $lmdir/lm_input
wait

# Next, for each type of language model, create the corresponding FST
# and the corresponding lang_test_* directory.

echo "Preparing language models for test"

# Create Ngram table
ngt -i=$lmdir/lm_input -n=$ngram -o=$lmdir/train${ngram}.ngt -b=yes
wait
# Estimate trigram and quadrigram models in ARPA format
tlm -tr=$lmdir/train${ngram}.ngt -n=$ngram -lm=wb -o=$lmdir/train${ngram}.arpa
wait



test=data/lang_test_${lm_suffix}

mkdir -p $test
cp -r $srcdir/* $test

cat $lmdir/train${ngram}.arpa | \
  arpa2fst --disambig-symbol=#0 \
           --read-symbol-table=$test/words.txt - $test/G.fst

utils/validate_lang.pl $test || exit 1;

echo "Succeeded in formatting data."
exit 0;
#rm -rf $tmpdir
#rm -f $ccs
