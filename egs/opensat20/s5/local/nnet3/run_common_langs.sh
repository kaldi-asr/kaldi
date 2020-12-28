#!/usr/bin/env bash

# Copyright 2016 Pegah Ghahremani

# This script used to generate MFCC+pitch features for input language lang.

. ./cmd.sh
set -e
stage=1
train_stage=-10
feat_suffix=_hires  # feature suffix for training data
echo "$0 $@"
[ -f conf/local.conf ] && . ./conf/local.conf
. ./utils/parse_options.sh

lang=$1
train_set=train

if [ $# -ne 1 ]; then
  echo "Usage:$0 [options] <language-id>"
  echo "e.g. $0 102-assamese"
  exit 1;
fi

if [ $stage -le 1 ]; then
  for datadir in train; do
    utils/data/perturb_data_dir_speed_3way.sh data/$lang/${datadir} data/$lang/${datadir}_sp
    steps/make_mfcc.sh --cmd "$train_cmd" --nj 70 data/$lang/${datadir}_sp
    steps/compute_cmvn_stats.sh data/$lang/${datadir}_sp
    utils/fix_data_dir.sh data/$lang/${datadir}_sp
  done
fi

train_set=train_sp
if [ $stage -le 2 ]; then
  steps/align_fmllr.sh \
    --nj 70 --cmd "$train_cmd" \
    data/$lang/$train_set data/$lang/lang_nosp_test exp/$lang/tri3 exp/$lang/tri3_ali_sp || exit 1
  touch exp/$lang/tri3_ali_sp/.done
fi

if [ $stage -le 3 ]; then
  for dataset in $train_set ; do
    utils/copy_data_dir.sh data/$lang/$dataset data/$lang/${dataset}_hires
    utils/data/perturb_data_dir_volume.sh data/$lang/${dataset}_hires
    steps/make_mfcc.sh --nj 70 --mfcc-config conf/mfcc_hires.conf --cmd "$train_cmd" \
      data/$lang/${dataset}_hires
    steps/compute_cmvn_stats.sh data/$lang/${dataset}_hires
    utils/fix_data_dir.sh data/$lang/${dataset}_hires
  done
fi
exit 0;
