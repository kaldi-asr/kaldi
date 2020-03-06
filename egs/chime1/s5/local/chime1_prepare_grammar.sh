#!/usr/bin/env bash

# Copyright 2015  University of Sheffield (Author: Ning Ma)
# Apache 2.0.
#
# Scripts for preparing grammar for the GRID corpus (or CHiME 1)

echo "Preparing grammar for test"

. ./config.sh # Needed for REC_ROOT and WAV_ROOT

# Setup relevant folders
lang="$REC_ROOT/data/lang"
utils="utils"

# Create FST grammar for the GRID
grammar_cmd="local/create_chime1_grammar.pl"

$grammar_cmd | fstcompile --isymbols=$lang/words.txt --osymbols=$lang/words.txt \
  --keep_isymbols=false --keep_osymbols=false | fstarcsort --sort_type=ilabel \
  > $lang/G.fst || exit 1

# Draw the FST
#echo "fstdraw --isymbols=$lang/words.txt --osymbols=$lang/words.txt $lang/G.fst | dot -Tps > local/G.ps"

echo "--> Grammar preparation succeeded"
exit 0
