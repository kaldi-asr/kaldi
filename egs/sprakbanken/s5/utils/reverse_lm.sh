#!/bin/bash

# Copyright 2012  Brno University of Technology (Author: Mirko Hannemann)
# JHU (Author: Dan Povey)
# Apache 2.0

# configuration section
tmpdir=data/local/lm_tmp  # only for OOVs and checks
lexicon=data/local/lang_tmp.reverse/lexicon.txt # only for checks
# end config section

mkdir -p $tmpdir

echo "$0 $@"  # Print the command line for logging

[ -f ./path.sh ] && . ./path.sh; # source the path.
. parse_options.sh || exit 1;

if [ $# != 3 ]; then
   echo "Usage: utils/reverse_lm.sh [options] <arpa-gz-file> <lang-dir> <out-dir>"
   echo "e.g.: utils/reverse_lm.sh data/local/nist_lm/lm_tgpr_5k.arpa.gz data/lang.reverse data/lang_test_tgpr_5k.reverse"
   echo "... where files from <lang-dir> are copied into <out-dir>"
   echo "options:"
   echo " --lexicon <lexicon-file>   reversed lexicon (only for checks)"
   exit 1;
fi

lm=$1 # gzipped arpa file
langdir=$2
outdir=$3 # output directory

# create the corresponding FST for the language model
# and the corresponding lang_test_* directory.

echo Preparing reverse language model from $lm into $outdir
echo "Finding OOVs and strange silences"
mkdir -p $outdir
for f in phones.txt words.txt L.fst L_disambig.fst phones/; do
  cp -r $langdir/$f $outdir
done
gunzip -c $lm | utils/find_arpa_oovs.pl $outdir/words.txt  > $tmpdir/oovs.txt

# grep -v '<s> <s>' because the LM seems to have some strange and useless
# stuff in it with multiple <s>'s in the history.  Encountered some other similar
# things in a LM from Geoff.  Removing all "illegal" combinations of <s> and </s>,
# which are supposed to occur only at being/end of utt.  These can cause 
# determinization failures of CLG [ends up being epsilon cycles].
gunzip -c $lm | \
  grep -v '<s> <s>' | \
  grep -v '</s> <s>' | \
  grep -v '</s> </s>' > $outdir/forward.arpa
echo "Mapping ARPA to reverse ARPA"
python utils/reverse_arpa.py $outdir/forward.arpa > $outdir/reverse.arpa
arpa2fst $outdir/reverse.arpa | fstprint | \
  grep -v "230258.5" | \
  utils/remove_oovs.pl $tmpdir/oovs.txt | \
  utils/eps2disambig.pl | utils/s2eps.pl | fstcompile --isymbols=$outdir/words.txt \
    --osymbols=$outdir/words.txt  --keep_isymbols=false --keep_osymbols=false \
    | fstrmepsilon > $outdir/G_org.fst
#--arc_type=log

echo "Push weights to make it stochastic (log semi-ring)"
# delta must be very small otherwise weight pushing won't succeed
#fstpush --push_weights=true --push_labels=true --delta=1E-7 $outdir/G_log.fst >$outdir/G_log_pushed.fst
fstpushspecial --delta=1E-5 $outdir/G_org.fst >$outdir/G.fst

fstisstochastic $outdir/G.fst
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

if [ -f $lexicon ]; then
  mkdir -p $tmpdir/g
  awk '{if(NF==1){ printf("0 0 %s %s\n", $1,$1); }} END{print "0 0 #0 #0"; print "0";}' \
    < "$lexicon"  >$tmpdir/g/select_empty.fst.txt
  fstcompile --isymbols=$outdir/words.txt --osymbols=$outdir/words.txt $tmpdir/g/select_empty.fst.txt | \
    fstarcsort --sort_type=olabel | fstcompose - $outdir/G.fst > $tmpdir/g/empty_words.fst
  fstinfo $tmpdir/g/empty_words.fst | grep cyclic | grep -w 'y' && 
  echo "Language model has cycles with empty words" && exit 1
  rm -r $tmpdir/g
fi
echo "Succeeded in creating reversed language model."
rm -r $tmpdir
