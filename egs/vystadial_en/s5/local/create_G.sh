#!/bin/bash

# Copyright 2012 Vassil Panayotov
#           2013 Ondrej Platek
# Apache 2.0

echo "===test_sets Formating data ..."
langdir=$1; shift
LMs=$1; shift
lmdir=$1; shift
lexicon=$1; shift

# Next, for each type of language model, create the corresponding FST
# and the corresponding lang_test_* directory.
for lm in $LMs ; do
    tgt=${langdir}_`basename "$lm"`
    lmp=$lmdir/`basename $lm`

    tmpdir=$tgt/tmp
    mkdir -p $tgt 
    mkdir -p $tmpdir

    echo "--- Preparing the grammar transducer (G.fst) from $lmp in $tgt ..."

    for f in phones.txt words.txt phones.txt L.fst L_disambig.fst phones ; do
        ln -s $langdir/$f $tgt/$f 2> /dev/null
    done

    cat $lmp | utils/find_arpa_oovs.pl $tgt/words.txt > $tmpdir/oovs.txt

     # grep -v '<s> <s>' because the LM seems to have some strange and useless
     # stuff in it with multiple <s>'s in the history.  Encountered some other similar
     # things in a LM from Geoff.  Removing all "illegal" combinations of <s> and </s>,
     # which are supposed to occur only at being/end of utt.  These can cause 
     # determinization failures of CLG [ends up being epsilon cycles].

    cat $lmp | \
      grep -v '<s> <s>\|</s> <s>\|</s> </s>' | \
      arpa2fst - | fstprint | \
      utils/remove_oovs.pl $tmpdir/oovs.txt | \
      utils/eps2disambig.pl | utils/s2eps.pl | fstcompile --isymbols=$tgt/words.txt \
        --osymbols=$tgt/words.txt  --keep_isymbols=false --keep_osymbols=false | \
      fstrmepsilon > $tgt/G.fst
    fstisstochastic $tgt/G.fst
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
    fstcompile --isymbols=$tgt/words.txt --osymbols=$tgt/words.txt \
      $tmpdir/g/select_empty.fst.txt | \
    fstarcsort --sort_type=olabel | fstcompose - $tgt/G.fst > $tmpdir/g/empty_words.fst
    fstinfo $tmpdir/g/empty_words.fst | grep cyclic | grep -w 'y' && 
      echo "Language model has cycles with empty words" && exit 1

    # rm -rf $tmpdir  # TODO debugging
    echo "*** Succeeded in creating G.fst for $tgt"

done  # for lm in $LMs ; do
