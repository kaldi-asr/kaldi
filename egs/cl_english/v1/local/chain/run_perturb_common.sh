#!/usr/bin/env bash

# Copyright 2021  Behavox (author: Hossein Hadian)
# Apache 2.0

set -euo pipefail

# This script:
# 1. Perturbs the train data and prepares hires features for training.
# 2. Prepares hires features for the test sets.
# 3. Appends _sp and _sp_hires suffixes accordingly.

# Note: this script does not overwrite.

stage=1
train_set=train_ldc
test_sets="test_ldc test_sp_oc"
nj=10
mfcc_hires_config=conf/mfcc_hires.conf
mfcc_config=conf/mfcc.conf
skip_hires_for_train=false
sp=true   # if false, will not speed perturb.

. ./cmd.sh
. ./path.sh
. utils/parse_options.sh

if [ $stage -le 1 ]; then
  echo "$0: Preparing low-resolution speed-perturbed data..."
  if [ -f data/${train_set}_sp/feats.scp ]; then
    echo "$0: data/${train_set}_sp/feats.scp already exits. Skipping this step..."
  else
    if $sp; then
      utils/data/perturb_data_dir_speed_3way.sh data/${train_set} data/${train_set}_sp
      echo "$0: making MFCC features for low-resolution speed-perturbed data"
      steps/make_mfcc.sh --mfcc-config $mfcc_config --cmd "$train_cmd" --nj $nj data/${train_set}_sp || exit 1;
      steps/compute_cmvn_stats.sh data/${train_set}_sp || exit 1;
      utils/fix_data_dir.sh data/${train_set}_sp
    else
      rm -r data/${train_set}_sp || true
      ln -sr data/${train_set} data/${train_set}_sp
    fi
  fi
fi

if [ $stage -le 2 ]; then
  echo "$0: Creating high-resolution MFCC features for test data."
  for datadir in ${test_sets}; do
      if [ -f data/${datadir}_hires/feats.scp ]; then
        echo "$0: data/${datadir}_hires/feats.scp already exits. Skipping this step..."
      else
        utils/copy_data_dir.sh data/$datadir data/${datadir}_hires
        steps/make_mfcc.sh --nj $nj --mfcc-config $mfcc_hires_config \
          --cmd "$train_cmd" data/${datadir}_hires || exit 1;
        steps/compute_cmvn_stats.sh data/${datadir}_hires || exit 1;
        utils/fix_data_dir.sh data/${datadir}_hires || exit 1;
      fi
  done
fi

if [ $stage -le 3 ] ; then
  echo "$0: Creating high-resolution MFCC features for train data."
  if [ -f data/${train_set}_sp_hires/feats.scp ]; then
    echo "$0: data/${train_set}_sp_hires/feats.scp already exits. Skipping this step..."
  elif $skip_hires_for_train; then
    printf "\n$0: skip_hires_for_train is true. Skipping this step...\n\n"
  else
    mfccdir=data/${train_set}_sp_hires/data
    datadir=${train_set}_sp
    utils/copy_data_dir.sh data/$datadir data/${datadir}_hires
    # do volume-perturbation on the training data prior to extracting hires
    # features; this helps make trained nnets more invariant to test data volume.
    utils/data/perturb_data_dir_volume.sh data/${train_set}_sp_hires || exit 1;
    steps/make_mfcc.sh --nj $nj --mfcc-config $mfcc_hires_config \
      --cmd "$train_cmd" data/${datadir}_hires || exit 1;
    steps/compute_cmvn_stats.sh data/${datadir}_hires || exit 1;
    utils/fix_data_dir.sh data/${datadir}_hires || exit 1;
  fi
fi

exit 0
