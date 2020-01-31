#!/usr/bin/env bash

# Make the features.

. ./cmd.sh
. ./path.sh
stage=1
set -e
. ./utils/parse_options.sh

mkdir -p exp/nnet2_online

if [ $stage -le 1 ]; then
  # this shows how you can split across multiple file-systems.  we'll split the
  # MFCC dir across multiple locations.  You might want to be careful here, if you
  # have multiple copies of Kaldi checked out and run the same recipe, not to let
  # them overwrite each other.
  mfccdir=mfcc
  utils/copy_data_dir.sh data/train_asr data/train_hires_asr
  steps/make_mfcc.sh --nj 10 --mfcc-config conf/mfcc_hires_16k.conf \
      --cmd "$train_cmd" data/train_hires_asr exp/make_hires/train $mfccdir || exit 1;
fi
