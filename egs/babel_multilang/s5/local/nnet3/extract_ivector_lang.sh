#!/usr/bin/env bash

# Copyright 2016 Pegah Ghahremani

# This scripts extracts iVector using global iVector extractor
# trained on all languages in multilingual setup.

. ./cmd.sh
set -e
stage=1
train_set=train_sp_hires # train_set used to extract ivector using shared ivector
                         # extractor.
ivector_suffix=_gb
nnet3_affix=

[ ! -f ./conf/common_vars.sh ] && echo 'the file conf/common_vars.sh does not exist!' && exit 1

. conf/common_vars.sh || exit 1;

. ./utils/parse_options.sh

lang=$1
global_extractor=$2

if [ $stage -le 7 ]; then
  # We extract iVectors on all the train_nodup data, which will be what we
  # train the system on.
  # having a larger number of speakers is helpful for generalization, and to
  # handle per-utterance decoding well (iVector starts at zero).
  utils/data/modify_speaker_info.sh --utts-per-spk-max 2 data/$lang/${train_set} data/$lang/${train_set}_max2
  if [ ! -f exp/$lang/nnet3${nnet3_affix}/ivectors_${train_set}${ivector_suffix}/ivector_online.scp ]; then
    steps/online/nnet2/extract_ivectors_online.sh --cmd "$train_cmd" --nj 200 \
      data/$lang/${train_set}_max2 $global_extractor exp/$lang/nnet3${nnet3_affix}/ivectors_${train_set}${ivector_suffix} || exit 1;
  fi
fi
exit 0;
