#!/bin/bash

# Copyright 2015, Mitsubishi Electric Research Laboratories, MERL (Author: Takaaki Hori)

# Config:
hidden=300 # Num-hidden units
class=200 # Num-classes
rnnlm_ver=rnnlm-0.3e # version of RNNLM to use
threads=1 # for RNNLM-HS
bptt=4 # length of BPTT unfolding in RNNLM
bptt_block=10 # length of BPTT unfolding in RNNLM

. utils/parse_options.sh || exit 1;

. ./path.sh
. ./cmd.sh ## You'll want to change cmd.sh to something that will work on your system.
           ## This relates to the queue.

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

# lm directories
dir=data/local/local_lm
srcdir=data/local/nist_lm
mkdir -p $dir

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
    }}' | sed "s/<UNK>/<RNN_UNK>/" > $dir/vocab_5k.rnn
else
  echo "Language model $srclm does not exist" && exit 1;
fi

# collect training data from WSJ0
touch $dir/train.rnn
if [ `du -m $dir/train.rnn | cut -f 1` -eq 223 ]; then
  echo "Not getting training data again [already exists]";
else
  echo "Collecting training data from $lm_train";
  gunzip -c $lm_train/{87,88,89}/*.z \
   | awk -v voc=$dir/vocab_5k.rnn '
   BEGIN{ while((getline<voc)>0) { invoc[$1]=1; }}
   /^</{next}{
     for (x=1;x<=NF;x++) { 
       w=toupper($x);
       if (invoc[w]) { printf("%s ",w); } else { printf("<RNN_UNK> "); }
     }
     printf("\n");
   }' > $dir/train.rnn
fi

# get validation data from CHiME3 dev set
touch $dir/valid.rnn
if [ `cat $dir/valid.rnn | wc -w` -eq 54239 ]; then
  echo "Not getting validation data again [already exists]";
else
  echo "Collecting validation data from $chime3_data/data/transcriptions";
  cut -d" " -f2- $chime3_data/data/transcriptions/dt05_real.trn_all \
                 $chime3_data/data/transcriptions/dt05_simu.trn_all \
      > $dir/valid.rnn
fi
  
# RNN language model traing
$KALDI_ROOT/tools/extras/check_for_rnnlm.sh "$rnnlm_ver" || exit 1

# train a RNN language model
rnnmodel=$dir/rnnlm_5k_h${hidden}_bptt${bptt}
if [ -f $rnnmodel ]; then
  echo "A RNN language model aready exists and is not constructed again"
  echo "To reconstruct, remove $rnnmodel first"
else
  echo "Training a RNN language model with $rnnlm_ver"
  echo "(runtime log is written to $dir/rnnlm.log)"
  $train_cmd $dir/rnnlm.log \
   $KALDI_ROOT/tools/$rnnlm_ver/rnnlm -train $dir/train.rnn -valid $dir/valid.rnn \
        -rnnlm $rnnmodel -hidden $hidden -class $class \
        -rand-seed 1 -independent -debug 1 -bptt $bptt -bptt-block $bptt_block || exit 1;
fi

# store in a RNNLM directory with necessary files
rnndir=data/lang_test_rnnlm_5k_h${hidden}
mkdir -p $rnndir
cp $rnnmodel $rnndir/rnnlm
grep -v -e "<s>" -e "</s>" $dir/vocab_5k.rnn > $rnndir/wordlist.rnn
touch $rnndir/unk.probs # make an empty file because we don't know unk-word probs.

