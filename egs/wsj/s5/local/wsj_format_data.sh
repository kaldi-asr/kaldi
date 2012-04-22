#!/bin/bash

# Copyright 2012  Microsoft Corporation  Daniel Povey
# Apache 2.0

# This script takes data prepared in a corpus-dependent way
# in data/local/, and converts it into the "canonical" form,
# in various subdirectories of data/, e.g. data/lang, data/lang_test_ug,
# data/train_si284, data/train_si84, etc.

# Don't bother doing train_si84 separately (although we have the file lists
# in data/local/) because it's just the first 7138 utterances in train_si284.
# We'll create train_si84 after doing the feature extraction.

. ./path.sh || exit 1;

echo "Preparing train and test data"

for x in train_si284 test_eval92 test_eval93 test_dev93 test_eval92_5k test_eval93_5k test_dev93_5k dev_dt_05 dev_dt_20; do 
  mkdir -p data/$x
  cp data/local/${x}_wav.scp data/$x/wav.scp || exit 1;
  cp data/local/$x.txt data/$x/text || exit 1;
  cp data/local/$x.spk2utt data/$x/spk2utt || exit 1;
  cp data/local/$x.utt2spk data/$x/utt2spk || exit 1;
  utils/filter_scp.pl data/$x/spk2utt data/local/spk2gender > data/$x/spk2gender || exit 1;
done

echo "Preparing word lists etc."

# Create the "lang" directory for training... we'll copy this same setup
# to be used in test too, and also add the G.fst.
# Note: the lexicon.txt and lexicon_disambig.txt are not used directly by
# the training scripts, so we put these in data/local/.

# TODO: make sure we properly handle the begin/end symbols in the lexicon,

mkdir -p data/lang


# (0), this is more data-preparation than data-formatting;
# add disambig symbols to the lexicon in data/local/lexicon.txt
# and produce data/local/lexicon_disambig.txt

ndisambig=`utils/add_lex_disambig.pl data/local/lexicon.txt data/local/lexicon_disambig.txt`
ndisambig=$[$ndisambig+1]; # add one disambig symbol for silence in lexicon FST.
echo $ndisambig > data/local/lex_ndisambig

# Format of lexicon_disambig.txt:
#!SIL    SIL
#<SPOKEN_NOISE>  SPN #1
#<UNK>   SPN #2
#<NOISE> NSN
#!EXCLAMATION-POINT      EH2_B K S K L AH0 M EY1 SH AH0 N P OY2 N T_E
#"CLOSE-QUOTE    K_B L OW1 Z K W OW1 T_E


# Create phones file with disambiguation symbols.
utils/add_disambig.pl --include-zero data/local/phones.txt \
  `cat data/local/lex_ndisambig` > data/lang/phones.txt

mkdir -p data/lang/phones/ # This will contain various sets of phones,
 # describing which are silence, which ones will be shared in the 
 # monophone system, etc., etc.


silphones="SIL SPN NSN"
for x in $silphones; do echo $x; done > data/lang/phones/silence.txt

grep -v '#' data/lang/phones.txt | grep -v -w -E `echo $silphones | sed 's/ /|/g'` | \
  grep -v '<eps>' | awk '{print $1}' > data/lang/phones/nonsilence.txt

grep '#' data/lang/phones.txt | awk '{print $1}' > data/lang/phones/disambig.txt

# Create these lists of phones in colon-separated integer list form too, 
# for purposes of being given to programs as command-line options.
for f in silence nonsilence disambig; do
  utils/sym2int.pl data/lang/phones.txt <data/lang/phones/$f.txt | \
   awk '{printf(":%d", $1);} END{printf "\n"}' | sed s/:// > data/lang/phones/$f.csl || exit 1;
done

cat data/local/lexicon.txt | awk '{print $1}' | sort | uniq  | \
 awk 'BEGIN{print "<eps> 0";} {printf("%s %d\n", $1, NR);} END{printf("#0 %d\n", NR+1);} ' \
  > data/lang/words.txt || exit 1;

# format of data/lang/words.txt:
#<eps> 0
#!EXCLAMATION-POINT 1
#!SIL 2
#"CLOSE-QUOTE 3
#...

# Create the basic L.fst without disambiguation symbols, for use
# in training. 
utils/make_lexicon_fst.pl data/local/lexicon.txt 0.5 SIL | \
  fstcompile --isymbols=data/lang/phones.txt --osymbols=data/lang/words.txt \
  --keep_isymbols=false --keep_osymbols=false | \
   fstarcsort --sort_type=olabel > data/lang/L.fst || exit 1;

# The file oov.txt contains a word that we will map any OOVs to during
# training.
echo "<SPOKEN_NOISE>" > data/lang/oov.txt || exit 1;

# (2)
# Create phonesets_*.txt and extra_questions.txt ...
# phonesets_mono.txt is sets of phones that are shared when building the monophone system
# and when asking questions based on an automatic clustering of phones, for the
# triphone system.  extra_questions.txt is some pre-defined extra questions about
# position and stress that split apart the categories we created in phonesets.txt.
# in extra_questions.txt there is also a question about silence phones, since we 
# don't include them in our automatically generated clustering of phones.

mkdir -p data/lang/phones

cat data/lang/phones/silence.txt | awk '{printf("%s ", $1);} END{printf "\n";}' \
  > data/lang/phones/sets_mono.txt || exit 1;

cat data/lang/phones/nonsilence.txt | \
  perl -e 'while(<>){ m:([A-Za-z]+)(\d*)(_.)?: || die "Bad line $_"; 
     $phone=$1; $stress=$2; $position=$3;
     if($phone eq $curphone){ print " $phone$stress$position"; }
  else { if(defined $curphone){ print "\n"; } $curphone=$phone;  print "$phone$stress$position";  }} print "\n"; ' \
 >> data/lang/phones/sets_mono.txt || exit 1;

grep -v -w `head -1 data/lang/phones/silence.txt` data/lang/phones/sets_mono.txt \
  > data/lang/phones/sets_cluster.txt || exit 1;

cat data/lang/phones/silence.txt | awk '{printf("%s ", $1);} END{printf "\n";}' \
  > data/lang/phones/extra_questions.txt
cat data/lang/phones/nonsilence.txt | perl -e 'while(<>){ m:([A-Za-z]+)(\d*)(_.)?: || die "Bad line $_"; 
     $phone=$1; $stress=$2; $pos=$3;
     $full_phone ="$1$2$3";
     $pos2list{$pos} = $pos2list{$pos} .  $full_phone . " ";
     $stress2list{$stress} = $stress2list{$stress} .  $full_phone . " ";
   } 
   foreach $k (keys %pos2list) { print "$pos2list{$k}\n"; } 
   foreach $k (keys %stress2list) { print "$stress2list{$k}\n"; }  ' \
 >> data/lang/phones/extra_questions.txt || exit 1;


( # Creating the "roots file" for building the context-dependent systems...
  # we share the roots across all the versions of each real phone.  We also
  # share the states of the 3 forms of silence.  "not-shared" here means the
  # states are distinct p.d.f.'s... normally we would automatically split on
  # the HMM-state but we're not making silences context dependent.
  echo "not-shared not-split $silphones";
  cat data/lang/phones/nonsilence.txt | \
    perl -e 'while(<>){ m:([A-Za-z]+)(\d*)(_.)?: || die "Bad line $_"; 
            $phone=$1; $stress=$2; $position=$3;
      if($phone eq $curphone){ print " $phone$stress$position"; }
      else { if(defined $curphone){ print "\n"; } $curphone=$phone; 
            print "shared split $phone$stress$position";  }} print "\n"; '
 ) > data/lang/phones/roots.txt || exit 1;

for x in sets_mono sets_cluster extra_questions; do
  utils/sym2int.pl data/lang/phones.txt <data/lang/phones/$x.txt > data/lang/phones/$x.int || exit 1;
done
utils/sym2int.pl --ignore-oov data/lang/phones.txt <data/lang/phones/roots.txt \
   > data/lang/phones/roots.int || exit 1;


silphonelist=`cat data/lang/phones/silence.csl | sed 's/:/ /g'`
nonsilphonelist=`cat data/lang/phones/nonsilence.csl | sed 's/:/ /g'`
cat conf/topo.proto | sed "s:NONSILENCEPHONES:$nonsilphonelist:" | \
   sed "s:SILENCEPHONES:$silphonelist:" > data/lang/topo


# Create the lexicon FST with disambiguation symbols, and put it in lang_test.
# There is an extra step where we create a loop to "pass through" the
# disambiguation symbols from G.fst.
phone_disambig_symbol=`grep \#0 data/lang/phones.txt | awk '{print $2}'`
word_disambig_symbol=`grep \#0 data/lang/words.txt | awk '{print $2}'`

utils/make_lexicon_fst.pl data/local/lexicon_disambig.txt 0.5 SIL '#'$ndisambig | \
   fstcompile --isymbols=data/lang/phones.txt --osymbols=data/lang/words.txt \
   --keep_isymbols=false --keep_osymbols=false |   \
   fstaddselfloops  "echo $phone_disambig_symbol |" "echo $word_disambig_symbol |" | \
   fstarcsort --sort_type=olabel > data/lang/L_disambig.fst || exit 1;


# Create L_align.fst, which is as L.fst but with alignment symbols (#1 and #2 at the
# beginning and end of words, on the input side)... useful if we
# ever need to e.g. create ctm's-- these are used to work out the
# word boundaries.
cat data/local/lexicon.txt | \
 awk '{printf("%s #1 ", $1); for (n=2; n <= NF; n++) { printf("%s ", $n); } print "#2"; }' | \
 utils/make_lexicon_fst.pl - 0.5 SIL | \
 fstcompile --isymbols=data/lang/phones.txt --osymbols=data/lang/words.txt \
  --keep_isymbols=false --keep_osymbols=false | \
 fstarcsort --sort_type=olabel > data/lang/L_align.fst

# Next, for each type of language model, create the corresponding FST
# and the corresponding lang_test_* directory.

echo Preparing language models for test

for lm_suffix in bg tgpr tg bg_5k tgpr_5k tg_5k; do
  test=data/lang_test_${lm_suffix}
  mkdir -p $test
  for f in phones.txt words.txt phones.txt L.fst L_disambig.fst \
     phones/; do
    cp -r data/lang/$f $test
  done
  gunzip -c data/local/lm_${lm_suffix}.arpa.gz | \
   utils/find_arpa_oovs.pl $test/words.txt  > data/local/oovs_${lm_suffix}.txt

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
    utils/remove_oovs.pl data/local/oovs_${lm_suffix}.txt | \
    utils/eps2disambig.pl | utils/s2eps.pl | fstcompile --isymbols=$test/words.txt \
      --osymbols=$test/words.txt  --keep_isymbols=false --keep_osymbols=false | \
     fstrmepsilon > $test/G.fst
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
