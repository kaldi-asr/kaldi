#!/bin/bash
# Copyright 2010-2011 Microsoft Corporation

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#  http://www.apache.org/licenses/LICENSE-2.0
#
# THIS CODE IS PROVIDED *AS IS* BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, EITHER EXPRESS OR IMPLIED, INCLUDING WITHOUT LIMITATION ANY IMPLIED
# WARRANTIES OR CONDITIONS OF TITLE, FITNESS FOR A PARTICULAR PURPOSE,
# MERCHANTABLITY OR NON-INFRINGEMENT.
# See the Apache 2 License for the specific language governing permissions and
# limitations under the License.


# The output of this script is the symbol tables data/{words.txt,phones.txt},
# and the grammars and lexicons data/{L,G}{,_disambig}.fst

# To be run from ..
if [ -f path.sh ]; then . path.sh; fi

cp data_prep/G.txt data/
scripts/make_words_symtab.pl < data/G.txt > data/words.txt
cp data_prep/lexicon.txt data/


scripts/make_phones_symtab.pl < data/lexicon.txt > data/phones.txt

silphones="sil"; # This would in general be a space-separated list of all silence phones.  E.g. "sil vn"
# Generate colon-separated lists of silence and non-silence phones.
scripts/silphones.pl data/phones.txt "$silphones" data/silphones.csl data/nonsilphones.csl

ndisambig=`scripts/add_lex_disambig.pl data/lexicon.txt data/lexicon_disambig.txt`
ndisambig=$[$ndisambig+1]; # add one disambig symbol for silence in lexicon FST.
scripts/add_disambig.pl data/phones.txt $ndisambig > data/phones_disambig.txt

# Create train transcripts in integer format:
cat data_prep/train_trans.txt | \
  scripts/sym2int.pl --ignore-first-field data/words.txt  > data/train.tra


# Get lexicon in FST format.

# silprob = 0.5: same prob as word.
scripts/make_lexicon_fst.pl data/lexicon.txt 0.5 sil  | fstcompile --isymbols=data/phones.txt --osymbols=data/words.txt --keep_isymbols=false --keep_osymbols=false | fstarcsort --sort_type=olabel > data/L.fst

scripts/make_lexicon_fst.pl data/lexicon_disambig.txt 0.5 sil '#'$ndisambig | fstcompile --isymbols=data/phones_disambig.txt --osymbols=data/words.txt --keep_isymbols=false --keep_osymbols=false | fstarcsort --sort_type=olabel > data/L_disambig.fst

fstcompile --isymbols=data/words.txt --osymbols=data/words.txt --keep_isymbols=false --keep_osymbols=false data/G.txt > data/G.fst

# Checking that G is stochastic [note, it wouldn't be for an Arpa]
fstisstochastic data/G.fst || echo Error


# Checking that disambiguated lexicon times G is determinizable
fsttablecompose data/L_disambig.fst data/G.fst | fstdeterminize >/dev/null || echo Error

# Checking that LG is stochastic:
fsttablecompose data/L.fst data/G.fst | fstisstochastic || echo Error

## Check lexicon.
## just have a look and make sure it seems sane.
fstprint   --isymbols=data/phones.txt --osymbols=data/words.txt data/L.fst  | head

