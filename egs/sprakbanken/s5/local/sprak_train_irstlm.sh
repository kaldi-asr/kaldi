#!/bin/bash

# Copyright 2013-2014  Mirsk Digital Aps (Author: Andreas Kirkedal)
# Apache 2.0

# This script takes data prepared in a corpus-dependent way
# in data/local/, and converts it into the "canonical" form,
# in various subdirectories of data/, e.g. data/lang, data/lang_test_ug,
# data/train_si284, data/train_si84, etc.

. ./path.sh || exit 1;
export PATH=$KALDI_ROOT/tools/irstlm/bin:$PATH

srcdict=$1
newtext=$2
lm_suffix=$3
N=$4
lmdir=$5
extdict=${srcdict}_$lm_suffix
tmpdir=data/local/lm_tmp
lang_tmp=data/local/lang_tmp
extlang=data/lang_$lm_suffix

if [ ! -d $lmdir ];
  then
  mkdir -p $lmdir
fi


if [ ! -d $extdict ];
  then

  echo "Creating $extdict based on $srcdict"

  # Extend the $srcdict to include the new data
  mkdir -p $extdict
  for f in extra_questions.txt lexicon.txt nonsilence_phones.txt optional_silence.txt silence_phones.txt; do
    cp $srcdict/$f $extdict/
  done

  mv $extdict/lexicon.txt $extdict/oldlexicon.txt
fi


if [ ! -f $extdict/transcripts.uniq ];
  then
  # Create the text data for LMs and RNNs
  cat $srcdict/transcripts.txt $newtext > $extdict/transcripts.txt
  sort -u $extdict/transcripts.txt > $extdict/transcripts.uniq
fi


if [ ! -f $extdict/lexicon.txt ];
  then
  # Extend lexicon with pronunciations from espeak
  echo "Transcibing $newtext using espeak"

  cat $newtext | tr [:blank:] '\n' | grep -P -v '^[\s?|\.|\!]*$' | sort -u > $extdict/wlist.txt

  # Piped so only a number is stored in the variable
  nwords=$(wc -l < $extdict/wlist.txt)
  nsplit=$((nwords / 8))

  # Create wordlist
  # Run through espeak to get phonetics
  split -l $nsplit $extdict/wlist.txt $extdict/Wtemp_
  for w in $extdict/Wtemp_*; do
    (cat $w | espeak -q -vda -x > $w.pho ) &
  done
  wait

  cat $extdict/Wtemp_*.pho > $extdict/plist.txt
  rm -f $extdict/Wtemp_*

  # Filter transcription
  # Remove diacritics, language annotation ((da), (en), (fr) etc.), insert space between symbols, remove 
  # initial and trailing spaces and collapse 2 or more spaces to one space
  cat $extdict/plist.txt | tr '^%,=:_|#$12;-?!' ' ' | tr "'" " " | perl -pe 's/\(..\)|\-|\~//g' | perl -pe 's// /g' | perl -pe 's/^ +| +$//g' | tr -s ' ' > $extdict/plist2.txt

  # Map phones with few occurences (Y, L, J, z, U, T, "Z" and x) to 
  # phones with many occurences (y, l, y, s, w, t, dZ and dZ respectively)
  cat $extdict/plist2.txt | tr 'BYLJzUT*Q' 'bylyswtRg' | perl -pe 's/d Z/dZ/g' | perl -pe 's/ ?x ?| Z ?|Z / dZ /g' > $extdict/plist3.txt

  # Create lexicon.txt and put it in data/local/dict
  paste $extdict/wlist.txt $extdict/plist3.txt > $extdict/lexicon1.txt

  # Remove entries without transcription
  grep -P  "^.+\t.+$" $extdict/lexicon1.txt > $extdict/newlexicon.txt

  echo "Combining lexicons"
  # Combine lexicons
  cat $extdict/oldlexicon.txt $extdict/newlexicon.txt > $extdict/templex
  sort -u $extdict/templex > $extdict/lexicon.txt
fi


if [ ! -d $extlang ];
  then
  # Create new lang_ext dir
  utils/prepare_lang.sh $extdict "<UNK>" $lang_tmp $extlang || exit 1;
fi

if [ ! -f $lmdir/extra4.ngt ];
  then
  echo "Preparing LM data"

  grep -P -v '^[\s?|\.|\!]*$' $newtext | \
  awk '{if(NF>=4){ printf("%s\n",$0); }}' > $lmdir/text.filt
    
  # Envelop LM training data in context cues
  $irstbin/add-start-end.sh < $lmdir/text.filt > $lmdir/lm_input


    echo "Creating new binary ngram table $lmdir/extra4.ngt"
    ngt -i=$lmdir/lm_input -n=4 -o=$lmdir/extra4.ngt -b=yes
fi

echo "Training ARPA model extra$lm_suffix"

# Randomly chose n=4 as upper bound for the ngram table
tlm -tr=$lmdir/extra4.ngt -n=$N -lm=wb -o=$lmdir/extra${N}$lm_suffix

# Next, create the corresponding FST
# and the corresponding lang_test_* directory.
test=data/lang_test_${N}${lm_suffix}
mkdir -p $test

for f in phones.txt words.txt phones.txt L.fst L_disambig.fst \
   phones/; do
  cp -r $extlang/$f $test
done

cat $lmdir/extra${N}$lm_suffix | \
utils/find_arpa_oovs.pl $test/words.txt  > $lmdir/oovs_${lm_suffix}.txt

  # grep -v '<s> <s>' because the LM seems to have some strange and useless
  # stuff in it with multiple <s>'s in the history.  Encountered some other similar
  # things in a LM from Geoff.  Removing all "illegal" combinations of <s> and </s>,
  # which are supposed to occur only at being/end of utt.  These can cause 
  # determinization failures of CLG [ends up being epsilon cycles].
cat $lmdir/extra${N}$lm_suffix | \
  grep -v '<s> <s>' | \
  grep -v '</s> <s>' | \
  grep -v '</s> </s>' | \
  arpa2fst - | fstprint | \
  utils/remove_oovs.pl $lmdir/oovs_${lm_suffix}.txt | \
  utils/eps2disambig.pl | utils/s2eps.pl | fstcompile --isymbols=$test/words.txt \
    --osymbols=$test/words.txt  --keep_isymbols=false --keep_osymbols=false | \
   fstrmepsilon > $test/G.fst
fstisstochastic $test/G.fst

echo "Succeeded in formatting data."
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
echo "Running diagnostics. Investigate if the LM has cycles."

mkdir -p $tmpdir
awk '{if(NF==1){ printf("0 0 %s %s\n", $1,$1); }} END{print "0 0 #0 #0"; print "0";}' \
  < "$lmdir/text.filt"  >$tmpdir/select_empty.fst.txt
fstcompile --isymbols=$test/words.txt --osymbols=$test/words.txt $tmpdir/select_empty.fst.txt | \
 fstarcsort --sort_type=olabel | fstcompose - $test/G.fst > $tmpdir/empty_words.fst
fstinfo $tmpdir/empty_words.fst | grep cyclic | grep -w 'y' && 
  echo "Language model has cycles with empty words" && exit 1


rm -rf $tmpdir
