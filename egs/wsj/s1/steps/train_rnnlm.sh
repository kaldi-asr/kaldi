#!/bin/bash

if [ $# != 3 ]; then
   echo "Usage: steps/train_rnnlm.sh <#hidden> <#class> <rnn-outfile-name>"
   echo "This is going to take a long time, try it first with "
   echo "steps/train_rnnlm.sh 5 140 test"
   echo "but to improve over baseline you'd need e.g. 350 hidden, 200 classes"
   exit 1;
fi

NumHidden=$1
NumClass=$2
ModelName=$PWD/$(basename $3)

TMP=$(mktemp)
rm -rf $TMP
mkdir -p $TMP

. path.sh || exit 1;

echo "Using $TMP"

echo "Extracting word symbols from baseline ngram LM"
gzcat data_prep/lm_bg.arpa.gz | awk '/^\\data/{dump=1;next}{if (dump)print}' | awk '/1-grams/{dump=1}/2-grams/{dump=0}{if (dump)print $2}' | awk '$0!=""{print}' > $TMP/vocab
cd $TMP

echo "##DAN - TODO! Copy these data somehow from the LCD std DVD set!!! "
echo "##(So far, I copy them locally from Brno :)"
echo "Using 99% for training, 1% for validation,limiting vocab to 20k word list"
gzcat /mnt/matylda2/data/WSJ1/13-32.1/wsj1/doc/lng_modl/lm_train/np_data/{87,88,89}/*  | awk '
BEGIN{
  while (getline<"vocab")v[$1]=$2
}
/^</{next}
{
  $0=toupper($0);
  for (i=1;i<=NF;i++) 
    if ($i in v)printf $i" "
    else printf "<UNK> ";
  print "";
}' | sed 's/ $//' > WSJ.txt

cat WSJ.txt | awk 'FNR%100==0{print $0}' > WSJ.valid
cat WSJ.txt | awk 'FNR%100!=0{print $0}' > WSJ.train

echo "Downloading latest version of rnnlm"
wget http://www.fit.vutbr.cz/~imikolov/rnnlm/rnnlm-0.3b.tgz
tar xzf rnnlm-0.3b.tgz
cd rnnlm-0.3b
g++ -lm -O2 -funroll-loops -fprefetch-loop-arrays -g  rnnlmlib.cpp rnnlm.cpp -o ../rnnlm
cd ..

echo "Training a RNN language model ($ModelName) with $NumHidden hidden neurons and $NumClass classes"
time ./rnnlm -rnnlm rnn -train WSJ.train -valid WSJ.valid -hidden $NumHidden -class $NumClass -bptt 5 -bptt-block 10 -debug 2 -alpha 0.1
mv rnn $ModelName

echo "Deleting $TMP"
rm -rf $TMP
