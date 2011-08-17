#!/bin/bash
#
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

# To be run from one directory above this script.



if [ -f path.sh ]; then . path.sh; fi


data_list="train test_mar87 test_oct87 test_feb89 test_oct89 test_feb91 test_sep92"

for x in lang $data_list; do
  mkdir -p data/$x
done

# Copy stuff into its final location:



for x in $data_list; do
  cp data/local/$x.spk2utt data/$x/spk2utt || exit 1;
  cp data/local/$x.utt2spk data/$x/utt2spk || exit 1;
  cp data/local/${x}_wav.scp data/$x/wav.scp || exit 1;
  cp data/local/${x}_trans.txt data/$x/text || exit 1;
done



scripts/make_words_symtab.pl < data/local/G.txt > data/lang/words.txt
scripts/make_phones_symtab.pl < data/local/lexicon.txt > data/lang/phones.txt

silphones="sil"; # This would in general be a space-separated list of all silence phones.  E.g. "sil vn"
# Generate colon-separated lists of silence and non-silence phones.
scripts/silphones.pl data/lang/phones.txt "$silphones" data/lang/silphones.csl \
  data/lang/nonsilphones.csl

ndisambig=`scripts/add_lex_disambig.pl data/local/lexicon.txt data/local/lexicon_disambig.txt`
ndisambig=$[$ndisambig+1]; # add one disambig symbol for silence in lexicon FST.
scripts/add_disambig.pl data/lang/phones.txt $ndisambig > data/lang/phones_disambig.txt


silprob=0.5  # same prob as word
scripts/make_lexicon_fst.pl data/local/lexicon.txt $silprob sil  | \
  fstcompile --isymbols=data/lang/phones.txt --osymbols=data/lang/words.txt \
   --keep_isymbols=false --keep_osymbols=false | \
   fstarcsort --sort_type=olabel > data/lang/L.fst

scripts/make_lexicon_fst.pl data/local/lexicon_disambig.txt $silprob sil '#'$ndisambig | \
   fstcompile --isymbols=data/lang/phones_disambig.txt --osymbols=data/lang/words.txt \
   --keep_isymbols=false --keep_osymbols=false | fstarcsort --sort_type=olabel \
    > data/lang/L_disambig.fst

fstcompile --isymbols=data/lang/words.txt --osymbols=data/lang/words.txt --keep_isymbols=false \
    --keep_osymbols=false data/local/G.txt > data/lang/G.fst

# Checking that G is stochastic [note, it wouldn't be for an Arpa]
fstisstochastic data/lang/G.fst || echo Error: G is not stochastic

# Checking that G.fst is determinizable.
fstdeterminize data/lang/G.fst /dev/null || echo Error determinizing G.

# Checking that L_disambig.fst is determinizable.
fstdeterminize data/lang/L_disambig.fst /dev/null || echo Error determinizing L.

# Checking that disambiguated lexicon times G is determinizable
fsttablecompose data/lang/L_disambig.fst data/lang/G.fst | fstdeterminize >/dev/null || echo Error

# Checking that LG is stochastic:
fsttablecompose data/lang/L.fst data/lang/G.fst | fstisstochastic || echo Error: LG is not stochastic.

## Check lexicon.
## just have a look and make sure it seems sane.
echo "First few lines of lexicon FST:"
fstprint   --isymbols=data/lang/phones.txt --osymbols=data/lang/words.txt data/lang/L.fst  | head


silphonelist=`cat data/lang/silphones.csl | sed 's/:/ /g'`
nonsilphonelist=`cat data/lang/nonsilphones.csl | sed 's/:/ /g'`
cat conf/topo.proto | sed "s:NONSILENCEPHONES:$nonsilphonelist:" | \
   sed "s:SILENCEPHONES:$silphonelist:" > data/lang/topo 

echo RM_format_data succeeded.


