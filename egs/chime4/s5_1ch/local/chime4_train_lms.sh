#!/bin/bash

# Modified from the script for CHiME3 baseline
# Copyright 2015, Mitsubishi Electric Research Laboratories, MERL (Author: Takaaki Hori)

# Config:
order=5 # n-gram order

. utils/parse_options.sh || exit 1;

. ./path.sh

if [ $# -ne 1 ]; then
  printf "\nUSAGE: %s <Chime4 root directory>\n\n" `basename $0`
  echo "Please specifies a Chime4 root directory"
  echo "If you use kaldi scripts distributed in the Chime4 data,"
  echo "It would be `pwd`/../.."
  exit 1;
fi

# check data directories
chime4_data=$1
wsj0_data=$chime4_data/data/WSJ0 # directory of WSJ0 in Chime4. You can also specify your WSJ0 corpus directory
if [ ! -d $chime4_data ]; then
  echo "$chime4_data does not exist. Please specify chime4 data root correctly" && exit 1
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
  ngram -lm $srclm -unk -map-unk '<UNK>' -write-vocab $dir/vocab_5k.txt
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

# get validation data from Chime4 dev set
touch $dir/valid.gz
if [ `du -k $dir/valid.gz | cut -f 1` -eq 68 ]; then
  echo "Not getting validation data again [already exists]";
else
  echo "Collecting validation data from $chime4_data/data/transcriptions";
  cut -d" " -f2- $chime4_data/data/transcriptions/dt05_real.trn_all \
                 $chime4_data/data/transcriptions/dt05_simu.trn_all \
      |gzip -c > $dir/valid.gz
fi

# train a large n-gram language model
lm_suffix=${order}gkn_5k
if [ -f $dir/lm_${lm_suffix}.arpa.gz ]; then
  echo "A $order-gram language model aready exists and is not constructed again"
  echo "To reconstruct, remove $dir/lm_${lm_suffix}.arpa.gz first"
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

# convert arpa LM to G.fst
echo "Converting the $order-gram language model to G.fst"
test=data/lang_test_${lm_suffix}
mkdir -p $test
cp -r data/lang/* $test || exit 1;

gunzip -c $dir/lm_${lm_suffix}.arpa.gz | \
    arpa2fst --disambig-symbol=#0 \
             --read-symbol-table=$test/words.txt - $test/G.fst

utils/validate_lang.pl --skip-determinization-check $test || exit 1;

echo "Succeeded in $order-gram LM training and conversion to G.fst"

