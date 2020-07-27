#!/usr/bin/env bash

# Copyright 2013-2014  Mirsk Digital Aps (Author: Andreas Kirkedal)
# Apache 2.0

# This script takes data prepared in a corpus-dependent way
# in data/local/, and converts it into the "canonical" form,
# in various subdirectories of data/, e.g. data/lang, data/lang_test_ug,
# data/train_si284, data/train_si84, etc.

. ./path.sh || exit 1;
if [ -z $IRSTLM ] ; then
  export IRSTLM=$KALDI_ROOT/tools/irstlm/
fi
export PATH=${PATH}:$IRSTLM/bin
if ! command -v ngt >/dev/null 2>&1 ; then
  echo "$0: Error: the IRSTLM is not available or compiled" >&2
  echo "$0: Error: We used to install it by default, but." >&2
  echo "$0: Error: this is no longer the case." >&2
  echo "$0: Error: To install it, go to $KALDI_ROOT/tools" >&2
  echo "$0: Error: and run extras/install_irstlm.sh" >&2
  exit 1
fi

srcdict=$1
newtext=$2
lm_suffix=$3
N=$4
lmdir=$5
extdict=${srcdict}_$lm_suffix
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


# Checks if espeak is available on the system. espeak is necessary to extend
# the setup because the original transcriptions were created with espeak and
# filtered

if ! which espeak >&/dev/null; then
  echo "espeak is not available on your system. You must install espeak before proceeding."
  exit 0;
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

  cat $dir/plist.txt | perl -pe 's/\([[a-z]{2}\)//g' | perl -pe 's// /g' | perl -pe 's/ a I / aI /g' | perl -pe 's/ d Z / dZ /g' | perl -pe 's/ \? / /g' | perl -pe 's/ ([\#]) /\+ /g' | perl -pe 's/([\@n3]) \- /\1\- /g' | perl -pe "s/[\_\:\!\'\,\|2]//g" | perl -pe 's/ \- / /g' | tr -s ' ' | perl -pe 's/^ +| +$//g' > $dir/plist2.txt

  #Some question marks are not caught above
  perl -pe 's/ \? / /g' $dir/plist2.txt > $dir/plist3.txt

  # Create lexicon.txt and put it in data/local/dict
  paste $dir/wlist.txt $dir/plist3.txt > $dir/lexicon1.txt

  # Remove entries without transcription
  grep -P  "^.+\t.+$" $dir/lexicon1.txt > $dir/lexicon2.txt

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
  add-start-end.sh < $lmdir/text.filt > $lmdir/lm_input


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


cp -r $extlang $test

cat $lmdir/extra${N}$lm_suffix | \
  arpa2fst --disambig-symbol=#0 \
           --read-symbol-table=$test/words.txt - $test/G.fst

utils/validate_lang.pl $test || exit 1;

exit 0;
