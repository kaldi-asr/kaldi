#!/bin/bash

# Apache 2.0.  Copyright 2013, Hong Kong University of Science and Technology (author: Ricky Chan Ho Yin)

# This script builds an interpolated LM from multiple seperated LMs produced from build_lm_cutoff_4gram.sh. 
# The interpolated LM is built by merging the seperated LMs with mixture weights - that minimizes perplexity on "test_file".
# Perplexities of the interpolated LM on "test_file" are test and write into file ${lm_name}/perplexity.txt

srilmbinpath=/homes/ricky/softwares/srilm/bin/i686-m64/ # You may need to modify the path

if [ $# -ne 3 ]
then
  echo
  echo usage:   $0 lm_name seperate_models_specifier test_file
  echo
  echo example: $0 mixlm123 model1+model2+model3 test_file
  echo
  exit
fi

NAME_LABEL=$1
MODELNAME=$2
MODELS=`echo $MODELNAME | awk '{gsub("+", " "); print $0}'`
TEST_FILE=$3


if [ ! -d $NAME_LABEL ]
then
 mkdir $NAME_LABEL
fi

echo "[[Initialization]]" 
for n in $MODELS
do
 $srilmbinpath/ngram -debug 2 -ppl $TEST_FILE -lm $n/lm_0/ug > $NAME_LABEL/$n.ug.prob
 $srilmbinpath/ngram -debug 2 -ppl $TEST_FILE -lm $n/lm_0/bg > $NAME_LABEL/$n.bg.prob
 $srilmbinpath/ngram -debug 2 -ppl $TEST_FILE -lm $n/lm_0/tg > $NAME_LABEL/$n.tg.prob
 $srilmbinpath/ngram -debug 2 -order 4 -ppl $TEST_FILE -lm $n/lm_0/fg > $NAME_LABEL/$n.fg.prob
done

echo "[[Compute weight]]" 
for n in ug bg tg fg
do
 argument=`echo $MODELS | awk -v aa=$NAME_LABEL -v tmp=$n '{for(i=1; i<=NF; i++) printf aa"/"$i "." tmp ".prob ";}'`
 $srilmbinpath/compute-best-mix $argument > $NAME_LABEL/$NAME_LABEL.$n.wgt
done

echo "[[Merging LM]]" 
for n in ug bg tg fg
do
  array_wgt=(`cat $NAME_LABEL/$NAME_LABEL.$n.wgt |cut -f2 -d\( | cut -f1 -d\)`)
  array_model=(`echo $MODELS`)
  len=`expr ${#array_model[*]} - 1`
  i=0
  COMMAND=`echo -lm ${array_model[$i]}/lm_0/$n -lambda ${array_wgt[$i]}`
  i=1
  while [ $i -lt $len ]
  do
    j=`expr $i + 1`
    COMMAND=`echo $COMMAND -mix-lm${j} ${array_model[$i]}/lm_0/$n -mix-lambda${j} ${array_wgt[$i]}`
    let i++
  done
  echo "$srilmbinpath/ngram -renorm -map-unk \<unk\> $COMMAND -mix-lm ${array_model[$i]}/lm_0/$n -write-lm $NAME_LABEL/$NAME_LABEL.$n"
  `echo $srilmbinpath/ngram -renorm -map-unk \<unk\> $COMMAND -mix-lm ${array_model[$i]}/lm_0/$n -write-lm $NAME_LABEL/$NAME_LABEL.$n`
done

echo "[[test perplexity]]" 
$srilmbinpath/ngram -lm $NAME_LABEL/$NAME_LABEL.ug -ppl $TEST_FILE > $NAME_LABEL/perplexity.txt
$srilmbinpath/ngram -lm $NAME_LABEL/$NAME_LABEL.bg -ppl $TEST_FILE >> $NAME_LABEL/perplexity.txt
$srilmbinpath/ngram -lm $NAME_LABEL/$NAME_LABEL.tg -ppl $TEST_FILE >> $NAME_LABEL/perplexity.txt
$srilmbinpath/ngram -lm $NAME_LABEL/$NAME_LABEL.fg -ppl $TEST_FILE >> $NAME_LABEL/perplexity.txt

