#!/bin/bash

if [ ! -d waves_yesno ]; then
  wget http://sourceforge.net/projects/kaldi/files/waves_yesno.tar.gz || exit 1;
  tar -xvzf waves_yesno.tar.gz || exit 1;
fi

train_base_name=train_yesno
test_base_name=test_yesno

rm -f -r data
rm -f -r exp 

# Data preparation
curdir=`pwd`
waves_dir=$curdir/waves_yesno
local/data_prep.sh ${waves_dir}
local/prepare_dict.sh
local/format_data.sh

# Feature extraction
mfccdir=$curdir/mfcc_yesno
for x in ${train_base_name} ${test_base_name}; do 
 steps/make_mfcc.sh data/$x exp/make_mfcc/$x $mfccdir 1
done

# Mono training
train_cmd="scripts/run.pl"
steps/train_mono.sh --num-jobs 1 --cmd "$train_cmd" \
  --start-gauss 30 --end-gauss 400 \
  data/${train_base_name} data/lang exp/mono0a 
  
# Graph compilation  
scripts/mkgraph.sh --mono data/lang_test_tg exp/mono0a exp/mono0a/graph_tgpr

# Decoding
decode_cmd="scripts/run.pl"
scripts/decode.sh --num-jobs 1 --cmd "$decode_cmd" --opts "--beam 10.0 --lattice-beam 2.0" \
   steps/decode_deltas.sh exp/mono0a/graph_tgpr data/${test_base_name} exp/mono0a/decode_${test_base_name}
