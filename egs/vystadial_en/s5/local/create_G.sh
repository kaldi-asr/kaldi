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

    cat $lmp | \
      arpa2fst --disambig-symbol=#0 \
               --read-symbol-table=$tgt/words.txt - $tgt/G.fst
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
