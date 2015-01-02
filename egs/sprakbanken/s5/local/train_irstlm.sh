#!/bin/bash

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

irstbin=$KALDI_ROOT/tools/irstlm/bin

#grep -P -v '^[\s?|\.|\!]*$' $lexicon | grep -v '^ *$' | \
#awk '{if(NF>=4){ printf("%s\n",$0); }}' > $lmdir/text.filt

# Envelop LM training data in context cues
$irstbin/add-start-end.sh < $lexicon | awk '{if(NF>=3){ printf("%s\n",$0); }}' > $lmdir/lm_input
wait

# Next, for each type of language model, create the corresponding FST
# and the corresponding lang_test_* directory.

echo "Preparing language models for test"

# Create Ngram table
$irstbin/ngt -i=$lmdir/lm_input -n=$ngram -o=$lmdir/train${ngram}.ngt -b=yes
wait
# Estimate trigram and quadrigram models in ARPA format
$irstbin/tlm -tr=$lmdir/train${ngram}.ngt -n=$ngram -lm=wb -o=$lmdir/train${ngram}.arpa
wait



test=data/lang_test_${lm_suffix}
mkdir -p $test

cp -rT $srcdir $test

cat $lmdir/train${ngram}.arpa | \
  utils/find_arpa_oovs.pl $test/words.txt  > $lmdir/oovs_${lm_suffix}.txt

  # grep -v '<s> <s>' because the LM seems to have some strange and useless
  # stuff in it with multiple <s>'s in the history.  Encountered some other similar
  # things in a LM from Geoff.  Removing all "illegal" combinations of <s> and </s>,
  # which are supposed to occur only at being/end of utt.  These can cause 
  # determinization failures of CLG [ends up being epsilon cycles].
cat $lmdir/train${ngram}.arpa | \
  grep -v '<s> <s>' | \
  grep -v '</s> <s>' | \
  grep -v '</s> </s>' | \
  arpa2fst - | fstprint | \
  utils/remove_oovs.pl $lmdir/oovs_${lm_suffix}.txt | \
  utils/eps2disambig.pl | utils/s2eps.pl | fstcompile --isymbols=$test/words.txt \
    --osymbols=$test/words.txt  --keep_isymbols=false --keep_osymbols=false | \
   fstrmepsilon | fstarcsort --sort_type=ilabel > $test/G.fst

utils/validate_lang.pl $test || exit 1;

echo "Succeeded in formatting data."
exit 0;
#rm -rf $tmpdir
#rm -f $ccs