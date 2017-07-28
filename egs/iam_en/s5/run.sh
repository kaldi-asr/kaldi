#!/bin/bash

stage=0
nj=20
variance_floor_val=0.1
. ./cmd.sh ## You'll want to change cmd.sh to something that will work on your system.
           ## This relates to the queue.
. utils/parse_options.sh  # e.g. this parses the --stage option if supplied.

if [ $stage -le 0 ]; then
  # data preparation
  local/prepare_data.sh --nj $nj
fi

mkdir -p data/{train,val_1,val_2,test}/data
mkdir -p char_data/{train,val_1,val_2,test}/data
if [ $stage -le 1 ]; then
  for f in train val_1 val_2 test; do
    local/make_feature_vect.py data/$f --scale-size 40 | \
      copy-feats --compress=true --compression-method=7 \
      ark:- ark,scp:data/$f/data/images.ark,data/$f/feats.scp || exit 1
  done
  steps/compute_cmvn_stats.sh data/train || exit 1;


  for f in train val_1 val_2 test; do
    local/make_feature_vect.py char_data/$f --scale-size 40 | \
      copy-feats --compress=true --compression-method=7 \
      ark:- ark,scp:char_data/$f/data/images.ark,char_data/$f/feats.scp || exit 1
  done
  steps/compute_cmvn_stats.sh char_data/train || exit 1;
fi

if [ $stage -le 2 ]; then
  local/prepare_dict.sh data/train/ data/dict
  utils/prepare_lang.sh --position-dependent-phones false data/dict "<sil>" data/lang/temp data/lang

  local/prepare_char_dict.sh char_data/train/ char_data/dict
  utils/prepare_lang.sh --position-dependent-phones false --sil-prob 0.0 char_data/dict "<sil>" char_data/lang/temp char_data/lang
fi

if [ $stage -le 3 ]; then
  ## Starting basic training on features
  steps/train_mono.sh --nj $nj \
    --variance_floor_val $variance_floor_val data/train data/lang exp/mono

  steps/train_mono.sh --nj $nj \
    --variance_floor_val $variance_floor_val char_data/train char_data/lang char_exp/mono
fi
