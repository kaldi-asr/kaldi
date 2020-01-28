#!/usr/bin/env bash

# Modified from the script for CHiME3 baseline
# Copyright 2015, Mitsubishi Electric Research Laboratories, MERL (Author: Takaaki Hori)

# Config:
order=5 # n-gram order

. utils/parse_options.sh || exit 1;

. ./path.sh

if [ $# -ne 1 ]; then
  printf "\nUSAGE: %s <CHiME3 root directory>\n\n" `basename $0`
  echo "Please specifies a CHiME3 root directory"
  echo "If you use kaldi scripts distributed in the CHiME3 data,"
  echo "It would be `pwd`/../.."
  exit 1;
fi

# check data directories
chime3_data=$1
wsj0_data=$chime3_data/data/WSJ0 # directory of WSJ0 in CHiME3. You can also specify your WSJ0 corpus directory
if [ ! -d $chime3_data ]; then
  echo "$chime3_data does not exist. Please specify chime3 data root correctly" && exit 1
fi
if [ ! -d $wsj0_data ]; then
  echo "$wsj0_data does not exist. Please specify WSJ0 corpus directory" && exit 1
fi
lm_train=$wsj0_data/wsj0/doc/lng_modl/lm_train/np_data

# check whether run_init is executed
if [ ! -d data/lang ]; then
  echo "error, execute local/run_init.sh, first"
  exit 1;
fi

# lm directories
dir=data/local/local_lm
srcdir=data/local/nist_lm
mkdir -p $dir

# check srilm ngram
! which ngram-count \
  && echo "SRILM tools not installed, which are required for LM training" && exit 1;

# extract 5k vocabulary from a baseline language model
srclm=$srcdir/lm_tgpr_5k.arpa.gz
if [ -f $srclm ]; then
  echo "Getting vocabulary from a baseline language model";
  gunzip -c $srclm | awk 'BEGIN{unig=0}{
    if(unig==0){
      if($1=="\\1-grams:"){unig=1}}
    else {
      if ($1 != "") {
        if ($1=="\\2-grams:" || $1=="\\end\\") {exit}
        else {print $2}}
    }}' > $dir/vocab_5k.txt
else
  echo "Language model $srclm does not exist" && exit 1;
fi

# collect training data from WSJ0
touch $dir/train.gz
if [ `du -m $dir/train.gz | cut -f 1` -eq 63 ]; then
  echo "Not getting training data again [already exists]";
else
  echo "Collecting training data from $lm_train";
  gunzip -c $lm_train/{87,88,89}/*.z \
   | awk -v voc=$dir/vocab_5k.txt '
   BEGIN{ while((getline<voc)>0) { invoc[$1]=1; }}
   /^</{next}{
     for (x=1;x<=NF;x++) {
       w=toupper($x);
       if (invoc[w]) { printf("%s ",w); } else { printf("<UNK> "); }
     }
     printf("\n");
   }' | gzip -c > $dir/train.gz
fi

# get validation data from CHiME3 dev set
touch $dir/valid.gz
if [ `du -k $dir/valid.gz | cut -f 1` -eq 68 ]; then
  echo "Not getting validation data again [already exists]";
else
  echo "Collecting validation data from $chime3_data/data/transcriptions";
  cut -d" " -f2- $chime3_data/data/transcriptions/dt05_real.trn_all \
                 $chime3_data/data/transcriptions/dt05_simu.trn_all \
      |gzip -c > $dir/valid.gz
fi

# train a large n-gram language model
lm_suffix=${order}gkn_5k
if [ -f $dir/lm_${lm_suffix}.arpa.gz ]; then
  echo "A $order-gram language model aready exists and is not constructed again"
  echo "To reconstruct, remove $dir/$dir/lm_${lm_suffix}.arpa.gz first"
else
  echo "Training a $order-gram language model"
  ngram-count -text $dir/train.gz -order $order \
              -vocab $dir/vocab_5k.txt -unk -map-unk "<UNK>" \
              -gt2min 1 -gt3min 1 -gt4min 2 -gt5min 2 \
              -interpolate -kndiscount \
              -lm $dir/lm_${lm_suffix}.arpa.gz
fi
echo "Checking validation perplexity of $order-gram language model"
ngram -order $order -ppl $dir/valid.gz -lm $dir/lm_${lm_suffix}.arpa.gz
# e.g. 5-gram perplexity:
# file data/local/local_lm/valid.txt: 3280 sentences, 54239 words, 3 OOVs
# 0 zeroprobs, logprob= -96775.5 ppl= 48.1486 ppl1= 60.8611

# Next, create the corresponding FST and lang_test_* directory.
echo "Preparing language models for test"
tmpdir=data/local/lm_tmp
lexicon=data/local/lang_tmp/lexiconp.txt
mkdir -p $tmpdir

test=data/lang_test_${lm_suffix}
mkdir -p $test
for f in phones.txt words.txt phones.txt L.fst L_disambig.fst \
   phones; do
  cp -r data/lang/$f $test
done
gunzip -c $dir/lm_${lm_suffix}.arpa.gz | \
  arpa2fst --disambig-symbol=#0 \
           --read-symbol-table=$test/words.txt - $test/G.fst
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
mkdir -p $tmpdir/g
awk '{if(NF==1){ printf("0 0 %s %s\n", $1,$1); }} END{print "0 0 #0 #0"; print "0";}' \
    < "$lexicon"  >$tmpdir/g/select_empty.fst.txt
fstcompile --isymbols=$test/words.txt --osymbols=$test/words.txt $tmpdir/g/select_empty.fst.txt | \
 fstarcsort --sort_type=olabel | fstcompose - $test/G.fst > $tmpdir/g/empty_words.fst
fstinfo $tmpdir/g/empty_words.fst | grep cyclic | grep -w 'y' &&
  echo "Language model has cycles with empty words" && exit 1
rm -r $tmpdir/g

echo "Succeeded in preparing a large ${order}-gram LM"
rm -r $tmpdir
