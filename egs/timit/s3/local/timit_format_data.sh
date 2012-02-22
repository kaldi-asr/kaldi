#!/bin/bash

# Copyright 2012  Navdeep Jaitly
# Copyright 2010-2011  Microsoft Corporation

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

arpa_lm=data/local/lm/biphone/lm_unpruned.gz

data_list="train test dev"

for x in lang lang_test $data_list; do
  mkdir -p data/$x
done

# Copy stuff into its final location:

for x in $data_list; do
  cp data/local/$x.spk2utt data/$x/spk2utt || exit 1;
  cp data/local/$x.utt2spk data/$x/utt2spk || exit 1;
  cp data/local/${x}_wav.scp data/$x/wav.scp || exit 1;
  cp data/local/${x}_trans.txt data/$x/text || exit 1;
  scripts/filter_scp.pl data/$x/spk2utt data/local/spk2gender.map > data/$x/spk2gender || exit 1;
done


scripts/make_words_symtab.pl < data/local/lexicon.txt > data/lang/words.txt
scripts/make_phones_symtab.pl < data/local/lexicon.txt > data/lang/phones.txt
cp data/lang/words.txt data/lang_test/words.txt

silphones="sil"; # This would in general be a space-separated list of all silence phones.  E.g. "sil vn"
# Generate colon-separated lists of silence and non-silence phones.
scripts/silphones.pl data/lang/phones.txt "$silphones" data/lang/silphones.csl \
  data/lang/nonsilphones.csl

ndisambig=`scripts/add_lex_disambig.pl data/local/lexicon.txt data/local/lexicon_disambig.txt`
ndisambig=$[$ndisambig+1]; # add one disambig symbol for silence in lexicon FST.
scripts/add_disambig.pl data/lang/phones.txt $ndisambig > data/lang_test/phones_disambig.txt
cp data/lang_test/phones_disambig.txt data/lang/ # needed for MMI.

echo "Creating L.fst"
silprob=0.3  # same prob as word
scripts/make_lexicon_fst.pl data/local/lexicon.txt $silprob sil  | \
  fstcompile --isymbols=data/lang/phones.txt --osymbols=data/lang/words.txt \
   --keep_isymbols=false --keep_osymbols=false | \
   fstarcsort --sort_type=olabel > data/lang/L.fst
echo "Done creating L.fst"


# L_disambig.fst has the disambiguation symbols (c.f. Mohri's papers)
echo "Creating L_disambig.fst"
scripts/make_lexicon_fst.pl data/local/lexicon_disambig.txt $silprob sil '#'$ndisambig | \
   fstcompile --isymbols=data/lang_test/phones_disambig.txt --osymbols=data/lang_test/words.txt \
   --keep_isymbols=false --keep_osymbols=false | fstarcsort --sort_type=olabel \
    > data/lang_test/L_disambig.fst
echo "Done creating L_disambig.fst"

cp data/lang_test/L_disambig.fst data/lang/  # Needed for MMI training.
echo "Creating G.fst"

#gunzip -c "$arpa_lm" | \
#   grep -v '<s> <s>' | \
#   grep -v '</s> <s>' | \
#   grep -v '</s> </s>' | \
#   arpa2fst - | fstprint | \
#   scripts/remove_oovs.pl /dev/null | \
#   scripts/eps2disambig.pl | scripts/s2eps.pl | \
#   fstcompile --isymbols=data/lang/words.txt --osymbols=data/lang_test/words.txt  --keep_isymbols=false \
#        --keep_osymbols=false > data/lang_test/G.fst
gunzip -c "$arpa_lm" | \
   grep -v '<s> <s>' | \
   grep -v '</s> <s>' | \
   grep -v '</s> </s>' | \
   arpa2fst - | fstprint | \
   scripts/remove_oovs.pl /dev/null | \
   scripts/s2eps.pl | \
   fstcompile --isymbols=data/lang/words.txt --osymbols=data/lang_test/words.txt  --keep_isymbols=false \
        --keep_osymbols=false > data/lang_test/G.fst

echo "G.fst created. How stochastic is it ?"
fstisstochastic data/lang_test/G.fst 

# Checking that G.fst is determinizable.
fstdeterminize data/lang_test/G.fst /dev/null || echo Error determinizing G.

# Checking that L_disambig.fst is determinizable.
fstdeterminize data/lang_test/L_disambig.fst /dev/null || echo Error determinizing L.

# Checking that disambiguated lexicon times G is determinizable
fsttablecompose data/lang_test/L_disambig.fst data/lang_test/G.fst | \
   fstdeterminize >/dev/null || echo Error

# Checking that LG is stochastic:
echo "How stochastic is LG.fst."
fstisstochastic data/lang_test/G.fst 
fsttablecompose data/lang/L.fst data/lang_test/G.fst | \
   fstisstochastic 

# Checking that LG_disambig.fst is stochastic:
echo "How stochastic is LG_disambig.fst."
fsttablecompose data/lang_test/L_disambig.fst data/lang_test/G.fst | \
   fstisstochastic 


## Check lexicon.
## just have a look and make sure it seems sane.
echo "First few lines of lexicon FST:"
fstprint   --isymbols=data/lang/phones.txt --osymbols=data/lang/words.txt data/lang/L.fst  | head


silphonelist=`cat data/lang/silphones.csl | sed 's/:/ /g'`
nonsilphonelist=`cat data/lang/nonsilphones.csl | sed 's/:/ /g'`
cat conf/topo.proto | sed "s:NONSILENCEPHONES:$nonsilphonelist:" | \
   sed "s:SILENCEPHONES:$silphonelist:" > data/lang/topo 

for x in phones.txt words.txt silphones.csl nonsilphones.csl topo; do
   cp data/lang/$x data/lang_test/$x || exit 1;
done

echo timit_format_data succeeded.
