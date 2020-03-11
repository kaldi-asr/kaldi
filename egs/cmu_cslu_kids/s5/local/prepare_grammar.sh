#!/usr/bin/env bash
#
# Copyright 2012  Johns Hopkins University (Author: Daniel Povey).  Apache 2.0.

# Takes no arguments. 



tmpdir=data/local/tmp
[ ! -f $tmpdir/G.txt ] && echo "No such file $tmpdir/G.txt" && exit 1;

. ./path.sh || exit 1; # for KALDI_ROOT

fstcompile --isymbols=data/lang/words.txt --osymbols=data/lang/words.txt --keep_isymbols=false \
    --keep_osymbols=false $tmpdir/G.txt | fstarcsort --sort_type=ilabel > data/lang/G.fst || exit 1;

# Checking that G is stochastic [note, it wouldn't be for an Arpa]
fstisstochastic data/lang/G.fst || echo Error: G is not stochastic

# Checking that G.fst is determinizable.
fstdeterminize data/lang/G.fst /dev/null || echo Error determinizing G.

# Checking that L_disambig.fst is determinizable.
fstdeterminize data/lang/L_disambig.fst /dev/null || echo Error determinizing L.

# Checking that disambiguated lexicon times G is determinizable
fsttablecompose data/lang/L_disambig.fst data/lang/G.fst | \
   fstdeterminize >/dev/null || echo Error

# Checking that LG is stochastic:
fsttablecompose data/lang/L.fst data/lang/G.fst | \
   fstisstochastic || echo Error: LG is not stochastic.

# Checking that L_disambig.G is stochastic:
fsttablecompose data/lang/L_disambig.fst data/lang/G.fst | \
   fstisstochastic || echo Error: LG is not stochastic.

echo "Succeeded preparing grammar for CMU_kids."
