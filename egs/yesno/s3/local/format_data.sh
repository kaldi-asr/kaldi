#!/bin/bash

. path.sh

echo "Preparing train and test data"

train_base_name=train_yesno
test_base_name=test_yesno

# This stage was copied from WSJ example
for x in ${train_base_name} ${test_base_name}; do 
  mkdir -p data/$x
  cp data/local/${x}_wav.scp data/$x/wav.scp
  cp data/local/$x.txt data/$x/text
  cat data/$x/text | awk '{printf("%s global\n", $1);}' > data/$x/utt2spk
  scripts/utt2spk_to_spk2utt.pl <data/$x/utt2spk >data/$x/spk2utt
done

echo "Preparing word lists etc."

# lang_test will contain common things we'll put in lang_test_{bg,tgpr,tg}
mkdir -p data/lang data/lang_test

# This stage was copied from WSJ example
# (0), this is more data-preparation than data-formatting;
# add disambig symbols to the lexicon in data/local/lexicon.txt
# and produce data/local/lexicon_disambig.txt
ndisambig=`scripts/add_lex_disambig.pl data/local/lexicon.txt data/local/lexicon_disambig.txt`
ndisambig=$[$ndisambig+1]; # add one disambig symbol for silence in lexicon FST.
echo $ndisambig > data/local/lex_ndisambig

# This stage was copied from WSJ example
# (1) Put into data/lang, phones.txt, silphones.csl, nonsilphones.csl, words.txt,
#   oov.txt
cp data/local/phones.txt data/lang # we could get these from the lexicon, but prefer to
 # do it like this so we get all the possible begin/middle/end versions of phones even
 # if they don't appear.  This is so if we extend the lexicon later, we could use the
 # same phones.txt (which is "baked into" the model and can't be changed after you build it).

silphones="SIL";
# Generate colon-separated lists of silence and non-silence phones.
scripts/silphones.pl data/lang/phones.txt "$silphones" data/lang/silphones.csl data/lang/nonsilphones.csl

cat data/local/lexicon.txt | awk '{print $1}' | sort | uniq  | \
 awk 'BEGIN{print "<eps> 0";} {printf("%s %d\n", $1, NR);} END{printf("#0 %d\n", NR+1);} ' \
  > data/lang/words.txt  
  

# Create the basic L.fst without disambiguation symbols, for use
# in training. 
scripts/make_lexicon_fst.pl data/local/lexicon.txt 0.999 SIL | \
  fstcompile --isymbols=data/lang/phones.txt --osymbols=data/lang/words.txt \
  --keep_isymbols=false --keep_osymbols=false | \
   fstarcsort --sort_type=olabel > data/lang/L.fst        

silphonelist=`cat data/lang/silphones.csl | sed 's/:/ /g'`
nonsilphonelist=`cat data/lang/nonsilphones.csl | sed 's/:/ /g'`
cat conf/topo.proto | sed "s:NONSILENCEPHONES:$nonsilphonelist:" | \
   sed "s:SILENCEPHONES:$silphonelist:" > data/lang/topo
  
echo "YES" > data/lang/oov.txt # No needed here, but expected by the scripts.

for f in phones.txt words.txt L.fst silphones.csl nonsilphones.csl topo oov.txt; do
  cp data/lang/$f data/lang_test
done


# (3),
# In lang_test, create a phones.txt file that includes the disambiguation symbols.
# the --include-zero includes the #0 symbol we pass through from the grammar.
# Note: we previously echoed the # of disambiguation symbols to data/local/lex_ndisambig.
scripts/add_disambig.pl --include-zero data/lang_test/phones.txt \
   `cat data/local/lex_ndisambig` > data/lang_test/phones_disambig.txt
   
cp data/lang_test/phones_disambig.txt data/lang # Needed for MMI.

# Create the lexicon FST with disambiguation symbols, and put it in lang_test.
# There is an extra
# step where we create a loop to "pass through" the disambiguation symbols
# from G.fst.  
phone_disambig_symbol=`grep \#0 data/lang_test/phones_disambig.txt | awk '{print $2}'`
word_disambig_symbol=`grep \#0 data/lang_test/words.txt | awk '{print $2}'`

scripts/make_lexicon_fst.pl data/local/lexicon_disambig.txt 0.999 SIL '#'$ndisambig | \
   fstcompile --isymbols=data/lang_test/phones_disambig.txt --osymbols=data/lang_test/words.txt \
   --keep_isymbols=false --keep_osymbols=false |   \
   fstaddselfloops  "echo $phone_disambig_symbol |" "echo $word_disambig_symbol |" | \
   fstarcsort --sort_type=olabel > data/lang_test/L_disambig.fst    
      
   
# Copy into data/lang/ also, where it will be needed for discriminative training.
cp data/lang_test/L_disambig.fst data/lang/

# Create L_align.fst, which is as L.fst but with alignment symbols (#1 and #2 at the
# beginning and end of words, on the input side)... useful if we
# ever need to e.g. create ctm's-- these are used to work out the
# word boundaries.

cat data/local/lexicon.txt | \
 awk '{printf("%s #0 ", $1); for (n=2; n <= NF; n++) { printf("%s ", $n); } print "#1"; }' | \
 scripts/make_lexicon_fst.pl - 0.999 SIL | \
 fstcompile --isymbols=data/lang_test/phones_disambig.txt --osymbols=data/lang_test/words.txt \
  --keep_isymbols=false --keep_osymbols=false | \
 fstarcsort --sort_type=olabel > data/lang_test/L_align.fst 
 
# Next, for each type of language model, create the corresponding FST
# and the corresponding lang_test directory.

echo Preparing language models for test

for lm_suffix in tg; do
  test=data/lang_test_${lm_suffix}
  mkdir -p $test
  for f in phones.txt words.txt phones_disambig.txt L.fst L_disambig.fst \
     silphones.csl nonsilphones.csl; do
    cp data/lang_test/$f $test
  done  

  
  # Original code was following:
  #gunzip -c data/local/lm_${lm_suffix}.arpa.gz | \
  #  grep -v '<s> <s>' | \
  #  grep -v '</s> <s>' | \
  #  grep -v '</s> </s>' | \
  #  arpa2fst - | fstprint | \
  #  scripts/remove_oovs.pl data/local/oovs_${lm_suffix}.txt | \
  #  scripts/eps2disambig.pl | scripts/s2eps.pl | fstcompile --isymbols=$test/words.txt \
  #    --osymbols=$test/words.txt  --keep_isymbols=false --keep_osymbols=false | \
  #   fstrmepsilon > $test/G.fst
  #fstisstochastic $test/G.fst   
  
  cat input/task.arpabo | arpa2fst - | fstprint | scripts/eps2disambig.pl | scripts/s2eps.pl | fstcompile --isymbols=$test/words.txt --osymbols=$test/words.txt --keep_isymbols=false --keep_osymbols=false | fstrmepsilon > $test/G.fst
  #cat input/G.txt | scripts/eps2disambig.pl | scripts/s2eps.pl | fstcompile --isymbols=$test/words.txt --osymbols=$test/words.txt --keep_isymbols=false --keep_osymbols=false | fstrmepsilon > $test/G.fst
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
