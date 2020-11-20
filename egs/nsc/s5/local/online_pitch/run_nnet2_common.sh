#!/usr/bin/env bash

# this script contains some common (shared) parts of the run_nnet*.sh scripts.

. ./cmd.sh


stage=0

set -e
. ./cmd.sh
. ./path.sh
. ./utils/parse_options.sh


if [ $stage -le 1 ]; then
  # Create high-resolution MFCC features (with 40 cepstra instead of 13).
  # this shows how you can split across multiple file-systems.  we'll split the
  # MFCC dir across multiple locations.  You might want to be careful here, if you
  # have multiple copies of Kaldi checked out and run the same recipe, not to let
  # them overwrite each other.
  mfccdir=mfcc
  if [[ $(hostname -f) == *.clsp.jhu.edu ]] && [ ! -d $mfccdir/storage ]; then
    utils/create_split_dir.pl /export/b0{1,2,3,4}/$USER/kaldi-data/egs/librispeech-$(date +'%m_%d_%H_%M')/s5/$mfccdir/storage $mfccdir/storage
  fi

  for datadir in train_960 dev_clean dev_other; do
    utils/copy_data_dir.sh data/$datadir data/${datadir}_hiresp
    steps/make_mfcc_pitch_online.sh --nj 150 --mfcc-config conf/mfcc_hires.conf \
       --online-pitch-config conf/online_pitch.conf \
      --cmd "$train_cmd" data/${datadir}_hiresp exp/make_hiresp/$datadir $mfccdir || exit 1;
    steps/compute_cmvn_stats.sh data/${datadir}_hiresp exp/make_hiresp/$datadir $mfccdir || exit 1;

    # dump plain MFCC features by selecting MFCC-only part
    steps/select_feats.sh 0-39 data/${datadir}_hiresp data/${datadir}_hires exp/make_hires/$datadir $mfccdir || exit 1;
    steps/compute_cmvn_stats.sh data/${datadir}_hires exp/make_hires/$datadir $mfccdir || exit 1;
  done

  # now create some data subsets.
  # mixed is the clean+other data.
  # 30k is 1/10 of the data (around 100 hours), 60k is 1/5th of it (around 200 hours).
  utils/subset_data_dir.sh data/train_960_hires 30000 data/train_mixed_hires_30k
  utils/subset_data_dir.sh data/train_960_hires 60000 data/train_mixed_hires_60k
fi

# The stages where we build the iVector extractor are the same as the
# non-pitch system, because the features given to the iVector extractor don't use pitch.
steps/online/run_nnet2_common.sh --stage 2 || exit 1;

exit 0;

