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

# To be run from ..
. path.sh

for lm_suffix in bg tg_pruned tg; do
  gunzip -c data_prep/lm_${lm_suffix}.arpa.gz | \
  scripts/find_arpa_oovs.pl data/words.txt  > data/oovs_${lm_suffix}.txt

  # grep -v '<s> <s>' because the LM seems to have some strange and useless
  # stuff in it with multiple <s>'s in the history.  Encountered some other similar
  # things in a LM from Geoff.  Removing all "illegal" combinations of <s> and </s>,
  # which are supposed to occur only at being/end of utt.  These can cause 
  # determinization failures of CLG.
  gunzip -c data_prep/lm_${lm_suffix}.arpa.gz | \
    grep -v '<s> <s>' | \
    grep -v '</s> <s>' | \
    grep -v '</s> </s>' | \
    arpa2fst - | fstprint | \
    scripts/remove_oovs.pl data/oovs_${lm_suffix}.txt | \
    scripts/eps2disambig.pl |  fstcompile --isymbols=data/words.txt --osymbols=data/words.txt \
     --keep_isymbols=false --keep_osymbols=false > data/G_${lm_suffix}.fst
  fstisstochastic data/G_${lm_suffix}.fst 
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
  mkdir -p tmpdir.g
  awk '{if(NF==1){ printf("0 0 %s %s\n", $1,$1); }} END{print "0 0 #0 #0"; print "0";}' \
    < data/lexicon.txt  >tmpdir.g/select_empty.fst.txt
  fstcompile --isymbols=data/words.txt --osymbols=data/words.txt tmpdir.g/select_empty.fst.txt | fstarcsort --sort_type=olabel  \
 | fstcompose - data/G_${lm_suffix}.fst > tmpdir.g/empty_words.fst
  fstinfo tmpdir.g/empty_words.fst | grep cyclic | grep -w 'y' && 
    echo "Language model has cycles with empty words" && exit 1
  rm -r tmpdir.g

done

