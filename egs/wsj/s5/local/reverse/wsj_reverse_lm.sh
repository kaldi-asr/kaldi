#!/bin/bash

# Copyright 2012  Brno University of Technology (Author: Mirko Hannemann)
# Apache 2.0

# configuration section
lmdir=data/local/nist_lm  # original arpa file
lm_suffix=tgpr_5k         # used for directory name
tmpdir=data/local/lm_tmp  # only for OOVs and checks
datadir=data/lang.reverse # for lexicon transducer and word lists
lexicon=data/local/lang_tmp.reverse/lexicon.txt # only for checks
lm=
# end config section

mkdir -p $tmpdir

echo "$0 $@"  # Print the command line for logging

[ -f ./path.sh ] && . ./path.sh; # source the path.
. parse_options.sh || exit 1;

if [ $# != 1 ]; then
   echo "Usage: local/reverse/wsj_reverse_lm.sh [options] <lm-suffix>"
   echo "... where <data-dir> is copied into the new decoding data dir"
   echo "e.g.: local/reverse/wsj_reverse_lm.sh tgpr_5k"
   echo "options:"
   echo " --lm <arpa-gz-file>        if non-standard lm is used"
   echo " --datadir <data-dir>       origin of lexicon transducer (data/lang.reverse/)"
   echo " --lexicon <lexicon-file>   reversed lexicon (only for checks)"
   exit 1;
fi

lm_suffix=$1
if [ -z "$lm" ]; then
  lm=$lmdir/lm_${lm_suffix}.arpa.gz
fi
test=data/lang_test_${lm_suffix}.reverse # output directory

# create the corresponding FST for the language model
# and the corresponding lang_test_* directory.

echo Preparing reverse language model from $lm into $test
echo "Finding OOVs and strange silences"
mkdir -p $test
for f in phones.txt words.txt L.fst L_disambig.fst phones/; do
  cp -r $datadir/$f $test
done
gunzip -c $lm | utils/find_arpa_oovs.pl $test/words.txt  > $tmpdir/oovs_${lm_suffix}.txt

# grep -v '<s> <s>' because the LM seems to have some strange and useless
# stuff in it with multiple <s>'s in the history.  Encountered some other similar
# things in a LM from Geoff.  Removing all "illegal" combinations of <s> and </s>,
# which are supposed to occur only at being/end of utt.  These can cause 
# determinization failures of CLG [ends up being epsilon cycles].
gunzip -c $lm | \
  grep -v '<s> <s>' | \
  grep -v '</s> <s>' | \
  grep -v '</s> </s>' > $test/forward.arpa
echo "Mapping ARPA to reverse ARPA"
python local/reverse_arpa.py $test/forward.arpa > $test/reverse.arpa
arpa2fst $test/reverse.arpa | fstprint | \
  grep -v "230258.5" | \
  utils/remove_oovs.pl $tmpdir/oovs_${lm_suffix}.txt | \
  utils/eps2disambig.pl | utils/s2eps.pl | fstcompile --isymbols=$test/words.txt \
    --osymbols=$test/words.txt  --keep_isymbols=false --keep_osymbols=false \
    | fstrmepsilon > $test/G_org.fst
#--arc_type=log

echo "Push weights to make it stochastic (log semi-ring)"
# delta must be very small otherwise weight pushing won't succeed
#fstpush --push_weights=true --push_labels=true --delta=1E-7 $test/G_log.fst >$test/G_log_pushed.fst
fstpushspecial --delta=1E-5 $test/G_org.fst >$test/G.fst

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
mkdir -p $tmpdir/g
awk '{if(NF==1){ printf("0 0 %s %s\n", $1,$1); }} END{print "0 0 #0 #0"; print "0";}' \
  < "$lexicon"  >$tmpdir/g/select_empty.fst.txt
fstcompile --isymbols=$test/words.txt --osymbols=$test/words.txt $tmpdir/g/select_empty.fst.txt | \
  fstarcsort --sort_type=olabel | fstcompose - $test/G.fst > $tmpdir/g/empty_words.fst
fstinfo $tmpdir/g/empty_words.fst | grep cyclic | grep -w 'y' && 
echo "Language model has cycles with empty words" && exit 1
rm -r $tmpdir/g

echo "Succeeded in creating reversed language model."
rm -r $tmpdir
