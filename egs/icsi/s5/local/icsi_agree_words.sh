#!/bin/bash

# Pawel Swietojanski, 2014
# Apache 2.0

#fix some issues with transcripts
#1) In case of e.g. ISN'T- remove '-' and check for ISN'T in a dict
#2) In case of e.g. ABO add '-' and check for ABO- in dict
#3) In case of e.g. AIR-CRAFT splice two words and check in dict or
#4) split AIR-CRAFT into two separate words

if [[ $# -ne 3  ]]; then
  echo 'Usage: <init-segments-file> <dict-file> <out-dir>';
  exit 1;
fi

trans=$1
dict=$2
odir=$3

wdir=$odir
mkdir -p $wdir

#make words lists
#if [[ ! -f $wdir/missing_words  ]]; then
  cut -d" " -f7- $trans | tr " " "\n" | sort -u > $wdir/training_words
  awk '{print $1}' $dict | sort -u > $wdir/dict_words
  comm -23 $wdir/training_words $wdir/dict_words > $wdir/missing_words
#fi

mw_cnt=$(wc -l $wdir/missing_words)
echo "Missing words $mw_cnt"

rm -f $wdir/words_to_replace
rm -f $wdir/words_to_remove
rm -f $wdir/words_to_add
rm -f $wdir/words_to_split

touch $wdir/words_to_replace
touch $wdir/words_to_remove
touch $wdir/words_to_add
touch $wdir/words_to_split

while read -r line; do
  if [[ $line =~ [A-Z\']+\-[A-Z\'\-]+$ ]]; then
     #echo "1: Words to split $line"
     word=`echo $line | sed -r "s!([A-Z']+)\-([A-Z']+)\-*!\1\2!"`
     #echo $word
     res=`grep -w "$word$" $wdir/dict_words | wc -l` 
     #echo $res
     if [[ $res -gt 0  ]]; then
        echo "$line $word" >> $wdir/words_to_replace
     else
        echo $line >> $wdir/words_to_split
     fi
  elif [[ $line =~ [A-Z\']+\-$ ]]; then
     #echo "2: Sign - to remove for $line"
     word=`basename $line -`
     res=`grep -w "$word$" $wdir/dict_words | wc -l` 
     #echo $res
     if [[ $res -gt 0  ]]; then
        echo "$line $word" >> $wdir/words_to_replace
     else
        echo "$line" >> $wdir/words_to_add
     fi
  elif [[ $line =~ [A-Z\']+$ ]]; then
     #echo "3: Sign - to add for $line"
     res=`grep -w "${line}-$" $wdir/dict_words | wc -l`
     #echo $res
     if [[ $res -gt 0  ]]; then
        echo "$line ${line}-" >> $wdir/words_to_replace
     else
        echo "$line" >> $wdir/words_to_add
     fi
  else
     #echo "4: Nothing to do with $line"
     echo $line >> $wdir/words_to_remove
  fi
done < $wdir/missing_words

cp $trans $wdir/segments0

local/icsi_agree_words_replace.pl $wdir/words_to_replace $wdir/segments0 $wdir/segments1
local/icsi_agree_words_split.pl $wdir/words_to_split $wdir/segments1 $wdir/segments2


