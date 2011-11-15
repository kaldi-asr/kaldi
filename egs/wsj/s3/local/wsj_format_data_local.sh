#!/bin/bash

# This script creates the appropriate lang_test directories for
# the locally created dictionary and lexicon.


# "bd" is big-dict.
# The "fgpr" LM is a locally estimated one (4-gram, pruned)
. path.sh || exit 1;
dict_srcdir=data/local/dict_larger_prep/
lm_srcdir=data/local/lm/4gram-mincount
lang=data/lang_test_bd_fgpr
lang_unpruned=data/lang_test_bd_fg
mkdir -p $lang

[ ! -f $dict_srcdir/lexicon.txt ] && \
   echo "First run wsj_prepare_local_dict.sh" && exit 1;
[ ! -f $lm_srcdir/lm_pr7.0.gz -o ! -f $lm_srcdir/lm_unpruned.gz ] && \
   echo "First run wsj_train_lms.sh" && exit 1;


# Get number of disambig symbols, and lexicon with disambig symbols.
ndisambig=`scripts/add_lex_disambig.pl $dict_srcdir/lexicon.txt $dict_srcdir/lexicon_disambig.txt`
ndisambig=$[$ndisambig+1]; # add one disambig symbol for silence in lexicon FST.
echo $ndisambig > $dict_srcdir/lex_ndisambig

# (1) Put into $lang, phones.txt, silphones.csl, nonsilphones.csl, words.txt,
#   oov.txt
cp data/local/phones.txt $lang

silphones="SIL SPN NSN";
# Generate colon-separated lists of silence and non-silence phones.
scripts/silphones.pl $lang/phones.txt "$silphones" $lang/silphones.csl $lang/nonsilphones.csl

# Make word symbol table.
cat $dict_srcdir/lexicon.txt | awk '{print $1}' | sort | uniq  | \
 awk 'BEGIN{print "<eps> 0";} {printf("%s %d\n", $1, NR);} END{printf("#0 %d\n", NR+1);} ' \
  > $lang/words.txt

# Create the basic L.fst without disambiguation symbols, for use
# in training. 
scripts/make_lexicon_fst.pl $dict_srcdir/lexicon.txt 0.5 SIL | \
  fstcompile --isymbols=$lang/phones.txt --osymbols=$lang/words.txt \
  --keep_isymbols=false --keep_osymbols=false | \
   fstarcsort --sort_type=olabel > $lang/L.fst

# The file oov.txt contains a word that we will map any OOVs to during
# training.  This won't be needed in a test directory, but for completion we
# do it.
echo "<SPOKEN_NOISE>" > data/lang/oov.txt

# Note: we don't need phonesets.txt and extra_questions.txt, as they are
# only needed during training.  So we don't bother creating them.
# Anyway they're the same as they would be in other lang directories.

silphonelist=`cat data/lang/silphones.csl | sed 's/:/ /g'`
nonsilphonelist=`cat data/lang/nonsilphones.csl | sed 's/:/ /g'`
cat conf/topo.proto | sed "s:NONSILENCEPHONES:$nonsilphonelist:" | \
   sed "s:SILENCEPHONES:$silphonelist:" > $lang/topo


# (3),
# In lang_test, create a phones.txt file that includes the disambiguation symbols.
# the --include-zero includes the #0 symbol we pass through from the grammar.
# Note: we previously echoed the # of disambiguation symbols to $dict_srcdir/lex_ndisambig.

scripts/add_disambig.pl --include-zero $lang/phones.txt \
   `cat $dict_srcdir/lex_ndisambig` > $lang/phones_disambig.txt

# Create the lexicon FST with disambiguation symbols, and put it in lang_test.
# There is an extra
# step where we create a loop to "pass through" the disambiguation symbols
# from G.fst.  
phone_disambig_symbol=`grep \#0 $lang/phones_disambig.txt | awk '{print $2}'`
word_disambig_symbol=`grep \#0 $lang/words.txt | awk '{print $2}'`

scripts/make_lexicon_fst.pl $dict_srcdir/lexicon_disambig.txt 0.5 SIL '#'$ndisambig | \
   fstcompile --isymbols=$lang/phones_disambig.txt --osymbols=$lang/words.txt \
   --keep_isymbols=false --keep_osymbols=false |   \
   fstaddselfloops  "echo $phone_disambig_symbol |" "echo $word_disambig_symbol |" | \
   fstarcsort --sort_type=olabel > $lang/L_disambig.fst || exit 1;


# Create L_align.fst, which is as L.fst but with alignment symbols (#1 and #2 at the
# beginning and end of words, on the input side)... useful if we
# ever need to e.g. create ctm's-- these are used to work out the
# word boundaries.
cat $dict_srcdir/lexicon.txt | \
 awk '{printf("%s #1 ", $1); for (n=2; n <= NF; n++) { printf("%s ", $n); } print "#2"; }' | \
 scripts/make_lexicon_fst.pl - 0.5 SIL | \
 fstcompile --isymbols=$lang/phones_disambig.txt --osymbols=$lang/words.txt \
  --keep_isymbols=false --keep_osymbols=false | \
 fstarcsort --sort_type=olabel > $lang/L_align.fst || exit 1;

# Next, for each type of language model, create the corresponding FST
# and the corresponding lang_test directory.

echo "Preparing language models for test"

# Note: at this point, $lang=="data/lang_test_bd_fgpr", we put a pruned 4-gram model
# there.

echo "Checking there are no OOVs" # there shouldn't be in this LM.
# If you have an LM with OOVs you'd have to put back the command
# "remove_oovs.pl" below, as it is used in wsj_format_data.sh.
gunzip -c $lm_srcdir/lm_pr7.0.gz | \
  scripts/find_arpa_oovs.pl $lang/words.txt | cmp - /dev/null || \
 exit 1;   


# Removing these "invalid combinations" of <s> and </s> is not 
# necessary because we produced these LMs ourselves, and they aren't
# broken.  But we'll leave this in the script just in case it gets modified
# later.
# Note: ~1.5M N-grams.
gunzip -c $lm_srcdir/lm_pr7.0.gz | \
  grep -v '<s> <s>' | \
  grep -v '</s> <s>' | \
  grep -v '</s> </s>' | \
  arpa2fst - | fstprint | \
    scripts/eps2disambig.pl | scripts/s2eps.pl | fstcompile --isymbols=$lang/words.txt \
      --osymbols=$lang/words.txt  --keep_isymbols=false --keep_osymbols=false | \
     fstrmepsilon > $lang/G.fst || exit 1;
  fstisstochastic $lang/G.fst

mkdir -p $lang_unpruned
cp $lang/* $lang_unpruned
# Be careful: this time we dispense with the grep -v '<s> <s>' so this might
# not work for LMs generated from all toolkits.
gunzip -c $lm_srcdir/lm_unpruned.gz | \
  arpa2fst - | fstprint | \
    scripts/eps2disambig.pl | scripts/s2eps.pl | fstcompile --isymbols=$lang/words.txt \
      --osymbols=$lang/words.txt  --keep_isymbols=false --keep_osymbols=false | \
     fstrmepsilon > $lang_unpruned/G.fst || exit 1;
  fstisstochastic $lang_unpruned/G.fst


# The commands below are just diagnostic tests.
 mkdir -p tmpdir.g
 awk '{if(NF==1){ printf("0 0 %s %s\n", $1,$1); }} END{print "0 0 #0 #0"; print "0";}' \
   < data/local/lexicon.txt  >tmpdir.g/select_empty.fst.txt
 fstcompile --isymbols=$lang/words.txt --osymbols=$lang/words.txt tmpdir.g/select_empty.fst.txt | \
  fstarcsort --sort_type=olabel | fstcompose - $lang/G.fst > tmpdir.g/empty_words.fst
 fstinfo tmpdir.g/empty_words.fst | grep cyclic | grep -w 'y' && 
   echo "Language model has cycles with empty words" && exit 1
  rm -r tmpdir.g




echo "Succeeded in formatting data."
