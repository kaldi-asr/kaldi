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

# This script takes data prepared in a corpus-dependent way
# in data/local/, and converts it into the "canonical" form,
# in various subdirectories of data/, e.g. data/lang/, data/lang_test_ug,
# data/train_si284, data/train_si84, etc.

# Don't bother doing train_si84 separately (although we have the file lists
# in data/local/) because it's just the first 7138 utterances in train_si284.
# We'll create train_si84 after doing the feature extraction.

. path.sh || exit 1;

echo "Preparing train and test data"

for x in train_si284 test_eval92 test_eval93 test_dev93; do 
  mkdir -p data/$x
  cp data/local/${x}_wav.scp data/$x/wav.scp
  cp data/local/$x.txt data/$x/text
  cp data/local/$x.spk2utt data/$x/spk2utt
  cp data/local/$x.utt2spk data/$x/utt2spk
  scripts/filter_scp.pl data/$x/spk2utt data/local/spk2gender.map > data/$x/spk2gender
done

echo "Preparing word lists etc."

# Create the "lang" directory for training... we'll copy this same setup
# to be used in test too, and also add the G.fst.
# Note: the lexicon.txt and lexicon_disambig.txt are not used directly by
# the training scripts, so we put these in data/local/.

# TODO: make sure we properly handle the begin/end symbols in the lexicon,

# lang_test will contain common things we'll put in lang_test_{bg,tgpr,tg}
mkdir -p data/lang data/lang_test


# (0), this is more data-preparation than data-formatting;
# add disambig symbols to the lexicon in data/local/lexicon.txt
# and produce data/local/lexicon_disambig.txt

ndisambig=`scripts/add_lex_disambig.pl data/local/lexicon.txt data/local/lexicon_disambig.txt`
ndisambig=$[$ndisambig+1]; # add one disambig symbol for silence in lexicon FST.
echo $ndisambig > data/local/lex_ndisambig


# (1) Put into data/lang, phones.txt, silphones.csl, nonsilphones.csl, words.txt,
#   oov.txt
cp data/local/phones.txt data/lang # we could get these from the lexicon, but prefer to
 # do it like this so we get all the possible begin/middle/end versions of phones even
 # if they don't appear.  This is so if we extend the lexicon later, we could use the
 # same phones.txt (which is "baked into" the model and can't be changed after you build it).

silphones="SIL SPN NSN";
# Generate colon-separated lists of silence and non-silence phones.
scripts/silphones.pl data/lang/phones.txt "$silphones" data/lang/silphones.csl data/lang/nonsilphones.csl

cat data/local/lexicon.txt | awk '{print $1}' | sort | uniq  | \
 awk 'BEGIN{print "<eps> 0";} {printf("%s %d\n", $1, NR);} END{printf("#0 %d\n", NR+1);} ' \
  > data/lang/words.txt

# Create the basic L.fst without disambiguation symbols, for use
# in training. 
scripts/make_lexicon_fst.pl data/local/lexicon.txt 0.5 SIL | \
  fstcompile --isymbols=data/lang/phones.txt --osymbols=data/lang/words.txt \
  --keep_isymbols=false --keep_osymbols=false | \
   fstarcsort --sort_type=olabel > data/lang/L.fst

# The file oov.txt contains a word that we will map any OOVs to during
# training.
echo "<SPOKEN_NOISE>" > data/lang/oov.txt

# (2)
# Create phonesets.txt and extra_questions.txt ...
# phonesets.txt is sets of phones that are shared when building the monophone system
# and when asking questions based on an automatic clustering of phones, for the
# triphone system.  extra_questions.txt is some pre-defined extra questions about
# position and stress that split apart the categories we created in phonesets.txt.
# in extra_questions.txt there is also a question about silence phones, since we 
# didn't include that in our

local/make_shared_phones.sh < data/lang/phones.txt > data/lang/phonesets_mono.txt
grep -v SIL data/lang/phonesets_mono.txt > data/lang/phonesets_cluster.txt
local/make_extra_questions.sh < data/lang/phones.txt > data/lang/extra_questions.txt

( # Creating the "roots file" for building the context-dependent systems...
  # we share the roots across all the versions of each real phone.  We also
  # share the states of the 3 forms of silence.  "not-shared" here means the
  # states are distinct p.d.f.'s... normally we would automatically split on
  # the HMM-state but we're not making silences context dependent.
  echo 'not-shared not-split SIL SPN NSN';
  cat data/lang/phones.txt | grep -v eps | grep -v SIL | grep -v SPN | grep -v NSN | awk '{print $1}' | \
    perl -e 'while(<>){ m:([A-Za-z]+)(\d*)(_.)?: || die "Bad line $_"; 
            $phone=$1; $stress=$2; $position=$3;
      if($phone eq $curphone){ print " $phone$stress$position"; }
      else { if(defined $curphone){ print "\n"; } $curphone=$phone; 
            print "shared split $phone$stress$position";  }} print "\n"; '
) > data/lang/roots.txt

silphonelist=`cat data/lang/silphones.csl | sed 's/:/ /g'`
nonsilphonelist=`cat data/lang/nonsilphones.csl | sed 's/:/ /g'`
cat conf/topo.proto | sed "s:NONSILENCEPHONES:$nonsilphonelist:" | \
   sed "s:SILENCEPHONES:$silphonelist:" > data/lang/topo

for f in phones.txt words.txt L.fst silphones.csl nonsilphones.csl topo; do
  cp data/lang/$f data/lang_test
done



# (3),
# In lang_test, create a phones.txt file that includes the disambiguation symbols.
# the --include-zero includes the #0 symbol we pass through from the grammar.
# Note: we previously echoed the # of disambiguation symbols to data/local/lex_ndisambig.
scripts/add_disambig.pl --include-zero data/lang_test/phones.txt \
   `cat data/local/lex_ndisambig` > data/lang_test/phones_disambig.txt


# Create the lexicon FST with disambiguation symbols, and put it in lang_test.
# There is an extra
# step where we create a loop "pass through" the disambiguation symbols
# from G.fst.  
phone_disambig_symbol=`grep \#0 data/lang_test/phones_disambig.txt | awk '{print $2}'`
word_disambig_symbol=`grep \#0 data/lang_test/words.txt | awk '{print $2}'`

scripts/make_lexicon_fst.pl data/local/lexicon_disambig.txt 0.5 SIL '#'$ndisambig | \
   fstcompile --isymbols=data/lang_test/phones_disambig.txt --osymbols=data/lang_test/words.txt \
   --keep_isymbols=false --keep_osymbols=false |   \
   fstaddselfloops  "echo $phone_disambig_symbol |" "echo $word_disambig_symbol |" | \
   fstarcsort --sort_type=olabel > data/lang_test/L_disambig.fst || exit 1;

cp data/lang_test/L_disambig.fst data/lang/


# Create L_align.fst, which is as L.fst but with alignment symbols (#1 and #2 at the
# beginning and end of words, on the input side)... useful if we
# ever need to e.g. create ctm's-- these are used to work out the
# word boundaries.
cat data/local/lexicon.txt | \
 awk '{printf("%s #1 ", $1); for (n=2; n <= NF; n++) { printf("%s ", $n); } print "#2"; }' | \
 scripts/make_lexicon_fst.pl - 0.5 SIL | \
 fstcompile --isymbols=data/lang_test/phones_disambig.txt --osymbols=data/lang_test/words.txt \
  --keep_isymbols=false --keep_osymbols=false | \
 fstarcsort --sort_type=olabel > data/lang_test/L_align.fst

# Next, for each type of language model, create the corresponding FST
# and the corresponding lang_test directory.

echo Preparing language models for test

for lm_suffix in bg tgpr tg; do
  test=data/lang_test_${lm_suffix}
  mkdir -p $test
  for f in phones.txt words.txt phones_disambig.txt L.fst L_disambig.fst \
     silphones.csl nonsilphones.csl; do
    cp data/lang_test/$f $test
  done
  gunzip -c data/local/lm_${lm_suffix}.arpa.gz | \
   scripts/find_arpa_oovs.pl $test/words.txt  > data/local/oovs_${lm_suffix}.txt

  # grep -v '<s> <s>' because the LM seems to have some strange and useless
  # stuff in it with multiple <s>'s in the history.  Encountered some other similar
  # things in a LM from Geoff.  Removing all "illegal" combinations of <s> and </s>,
  # which are supposed to occur only at being/end of utt.  These can cause 
  # determinization failures of CLG [ends up being epsilon cycles].
  gunzip -c data/local/lm_${lm_suffix}.arpa.gz | \
    grep -v '<s> <s>' | \
    grep -v '</s> <s>' | \
    grep -v '</s> </s>' | \
    arpa2fst - | fstprint | \
    scripts/remove_oovs.pl data/local/oovs_${lm_suffix}.txt | \
    scripts/eps2disambig.pl | scripts/s2eps.pl | fstcompile --isymbols=$test/words.txt \
      --osymbols=$test/words.txt  --keep_isymbols=false --keep_osymbols=false \
      > $test/G.fst
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
  mkdir -p tmpdir.g
  awk '{if(NF==1){ printf("0 0 %s %s\n", $1,$1); }} END{print "0 0 #0 #0"; print "0";}' \
    < data/local/lexicon.txt  >tmpdir.g/select_empty.fst.txt
  fstcompile --isymbols=$test/words.txt --osymbols=$test/words.txt tmpdir.g/select_empty.fst.txt | \
   fstarcsort --sort_type=olabel | fstcompose - $test/G.fst > tmpdir.g/empty_words.fst
  fstinfo tmpdir.g/empty_words.fst | grep cyclic | grep -w 'y' && 
    echo "Language model has cycles with empty words" && exit 1
  rm -r tmpdir.g
done

echo "Succeeded in formatting data."
