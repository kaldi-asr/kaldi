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

for f in phones.txt words.txt phones.txt L.fst L_disambig.fst \
   phones/; do
  cp -r data/lang/$f $test
done

cat $lmdir/sprak.arpa | \
utils/find_arpa_oovs.pl $test/words.txt  > $lmdir/oovs_${lm_suffix}.txt

  # grep -v '<s> <s>' because the LM seems to have some strange and useless
  # stuff in it with multiple <s>'s in the history.  Encountered some other similar
  # things in a LM from Geoff.  Removing all "illegal" combinations of <s> and </s>,
  # which are supposed to occur only at being/end of utt.  These can cause 
  # determinization failures of CLG [ends up being epsilon cycles].
cat $lmdir/sprak.arpa | \
  grep -v '<s> <s>' | \
  grep -v '</s> <s>' | \
  grep -v '</s> </s>' | \
  arpa2fst - | fstprint | \
  utils/remove_oovs.pl $lmdir/oovs_${lm_suffix}.txt | \
  utils/eps2disambig.pl | utils/s2eps.pl | fstcompile --isymbols=$test/words.txt \
    --osymbols=$test/words.txt  --keep_isymbols=false --keep_osymbols=false | \
   fstrmepsilon | fstarcsort --sort_type=ilabel > $test/G.fst
fstisstochastic $test/G.fst
 # The output is like:
 # 9.14233e-05 -0.259833
 # we do expect the first of these 2 numbers to be close to zero (the second is
 # nonzero because the backoff weights make the states sum to >1).
 # Because of the <s> fiasco for these particular LMs, the first number is not
 # as close to zero as it could be.

  # Everything below is only for diagnostic.
  # Checking that G has no cycles with empty words on them (e.g. <s>, </s>);
  # this might cause determinization failure of CLG.
  # #0 is treated as an empty word.
mkdir -p $tmpdir
awk '{if(NF==1){ printf("0 0 %s %s\n", $1,$1); }} END{print "0 0 #0 #0"; print "0";}' \
  < "$lexicon"  >$tmpdir/select_empty.fst.txt
fstcompile --isymbols=$test/words.txt --osymbols=$test/words.txt $tmpdir/select_empty.fst.txt | \
 fstarcsort --sort_type=olabel | fstcompose - $test/G.fst > $tmpdir/empty_words.fst
fstinfo $tmpdir/empty_words.fst | grep cyclic | grep -w 'y' && 
  echo "Language model has cycles with empty words" && exit 1

echo "Succeeded in formatting data."
rm -r $tmpdir
