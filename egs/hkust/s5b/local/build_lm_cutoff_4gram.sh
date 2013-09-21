#!/bin/bash

# Apache 2.0.  Copyright 2013, Hong Kong University of Science and Technology (author: Ricky Chan Ho Yin)
#                              Cambridge University Engineering Department Alumni

# This script builds individual languge model in arpa format using either SRILM or HTK binaries. 
# Arpa format LM can be applied to Kaldi speech recognition toolkit for training and decoding
#
# If --use_sri option is applied, SRILM training with witten-bell discounting is used, otherwise HTK 
# training is used.
#
# lm_name => LM name
# train_scp => file contains list of LM training text files
# wordlist => vocabulary list
# test_file => file for perplexity test
#
# htk_config => htk LM config
# bg_cutoff => bigram cutoff frequency threshold
# tg_cutoff => trigram cutoff frequency threshold
# fg_cutoff => fourgram cutoff frequency threshold


srilmbinpath=/homes/ricky/softwares/srilm/bin/i686-m64 # You probably need to modify the path 
htklmbinpath=/homes/ricky/softwares/htk/bin # You probably need to modify the path

if [ $# -ne 8 ] && [ $# -ne 5 ]; then 
  echo Usage: $0 lm_name --use_sri train_scp wordlist test_file 
  echo Usage: $0 lm_name htk_config train_scp wordlist test_file bg_cutoff tg_cutoff fg_cutoff
  exit;
fi

NAME_LABEL=$1
SCP=$3
WORDLIST=$4
TEST_FILE=$5
if [ $2 == "--use_sri" ]; then
  numfile=`wc $SCP | awk '{print $1}'`
  if [ $numfile -gt 1 ]; then
    tmpdir=$(mktemp -d)
    TRAIN_FILE=$tmpdir/train_text
    rm -f $TRAIN_FILE
    for n in `cat $SCP`; do 
      cat $n >> $TRAIN_FILE
    done
  else 
    TRAIN_FILE=`cat $SCP`
  fi
else
  CONFIG=$2
  BG_CUT=$6
  TG_CUT=$7
  FG_CUT=$8
fi

if [ ! -d $NAME_LABEL ]
then
 mkdir $NAME_LABEL
fi

LM_PLACE=$NAME_LABEL/lm_0
if [ ! -d $LM_PLACE ] 
then
 mkdir $LM_PLACE
fi

if [ $2 == "--use_sri" ]; then
  echo "[[build N-grams]]"
  $srilmbinpath/ngram-count -wbdiscount1 -order 1 -text $TRAIN_FILE -vocab $WORDLIST -lm $LM_PLACE/ug
  $srilmbinpath/ngram-count -wbdiscount1 -wbdiscount2 -order 2 -text $TRAIN_FILE -vocab $WORDLIST -lm $LM_PLACE/bg
  $srilmbinpath/ngram-count -wbdiscount1 -wbdiscount2 -wbdiscount3 -order 3 -text $TRAIN_FILE -vocab $WORDLIST -lm $LM_PLACE/tg
  $srilmbinpath/ngram-count -wbdiscount1 -wbdiscount2 -wbdiscount3 -wbdiscount4 -order 4 -text $TRAIN_FILE -vocab $WORDLIST -lm $LM_PLACE/fg

  echo "[[test perplexity]]"
  $srilmbinpath/ngram -order 1 -lm $LM_PLACE/ug -ppl $TEST_FILE > $NAME_LABEL/perplexity.txt
  $srilmbinpath/ngram -order 2 -lm $LM_PLACE/bg -ppl $TEST_FILE >> $NAME_LABEL/perplexity.txt
  $srilmbinpath/ngram -order 3 -lm $LM_PLACE/tg -ppl $TEST_FILE >> $NAME_LABEL/perplexity.txt
  $srilmbinpath/ngram -order 4 -lm $LM_PLACE/fg -ppl $TEST_FILE >> $NAME_LABEL/perplexity.txt

  rm -rf $tmpdir
  exit
fi

LM_PLACE2=$NAME_LABEL/lm_1
if [ ! -d $LM_PLACE2 ]
then
 mkdir $LM_PLACE2
fi

echo "[[Initialization]]" 
$htklmbinpath/LNewMap -f WFC $NAME_LABEL $NAME_LABEL/empty.wmap
$htklmbinpath/LGPrep -A -T 1 -a 5000000 -b 100000000 -d $NAME_LABEL -n 4 -S $SCP $NAME_LABEL/empty.wmap
$htklmbinpath/LGCopy -T 1 -n 4 -b 10000000 -d $LM_PLACE $NAME_LABEL/wmap $NAME_LABEL/gram.*
$htklmbinpath/LGCopy -T 4 -n 4 -A -C $CONFIG -m $LM_PLACE2/wmap -a 5000000 -b 10000000 -d $LM_PLACE2 -w $WORDLIST $NAME_LABEL/wmap $LM_PLACE/data.*
$htklmbinpath/LFoF -T 1 -A -n 4 -f 128 $LM_PLACE2/wmap $LM_PLACE2/lm.train.fof $LM_PLACE2/data.* $LM_PLACE2/data.*

echo "[[build ug]]" 
$htklmbinpath/LBuild -T 1 -A -f TEXT -n 1 -t $LM_PLACE2/lm.train.fof $LM_PLACE2/wmap $LM_PLACE/ug $LM_PLACE2/data.* $LM_PLACE2/data.*
echo "[[build bg]]" 
$htklmbinpath/LBuild -T 1 -A -f TEXT -n 2 -c 2 $BG_CUT -t $LM_PLACE2/lm.train.fof $LM_PLACE2/wmap $LM_PLACE/bg $LM_PLACE2/data.* $LM_PLACE2/data.*
echo "[[build tg]]" 
$htklmbinpath/LBuild -T 1 -A -f TEXT -n 3 -c 2 $BG_CUT -c 3 $TG_CUT -t $LM_PLACE2/lm.train.fof $LM_PLACE2/wmap $LM_PLACE/tg $LM_PLACE2/data.* $LM_PLACE2/data.*
echo "[[build fg]]" 
$htklmbinpath/LBuild -T 1 -A -f TEXT -n 4 -c 2 $BG_CUT -c 3 $TG_CUT -c 4 $FG_CUT -t $LM_PLACE2/lm.train.fof $LM_PLACE2/wmap $LM_PLACE/fg $LM_PLACE2/data.* $LM_PLACE2/data.*

echo "[[test perplexity]]" 
$htklmbinpath/LPlex -n 1 -C $CONFIG -t $LM_PLACE/ug $TEST_FILE > $NAME_LABEL/perplexity.txt
$htklmbinpath/LPlex -n 2 -C $CONFIG -t $LM_PLACE/bg $TEST_FILE >> $NAME_LABEL/perplexity.txt
$htklmbinpath/LPlex -n 3 -C $CONFIG -t $LM_PLACE/tg $TEST_FILE >> $NAME_LABEL/perplexity.txt
$htklmbinpath/LPlex -n 4 -C $CONFIG -t $LM_PLACE/fg $TEST_FILE >> $NAME_LABEL/perplexity.txt

