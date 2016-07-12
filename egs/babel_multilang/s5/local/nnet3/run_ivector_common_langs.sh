#!/bin/bash

. ./cmd.sh
set -e
stage=1
train_stage=-10
generate_alignments=true # false if doing ctc training
speed_perturb=true
pitch_conf=conf/pitch.conf
use_flp=false
voicing_conf=
global_extractor=exp/multi/nnet3/extractor
ivector_suffix=

[ ! -f ./conf/common_vars.sh ] && echo 'the file conf/common_vars.sh does not exist!' && exit 1

. conf/common_vars.sh || exit 1;

[ -f local.conf ] && . ./local.conf

. ./utils/parse_options.sh

L=$1

mkdir -p nnet3
# perturbed data preparation
train_set=train
if [ "$speed_perturb" == "true" ]; then
  train_set=train_sp
fi

extractor=$global_extractor
ivector_suffix=${ivector_suffix}_gb

if [ $stage -le 8 ]; then
  # We extract iVectors on all the train_nodup data, which will be what we
  # train the system on.
  
  # having a larger number of speakers is helpful for generalization, and to
  # handle per-utterance decoding well (iVector starts at zero).
  steps/online/nnet2/copy_data_dir.sh --utts-per-spk-max 2 data/$L/${train_set}_hires data/$L/${train_set}_max2_hires
  
  if [ ! -f exp/$L/nnet3/ivectors_${train_set}${ivector_suffix}/ivector_online.scp ]; then
    steps/online/nnet2/extract_ivectors_online.sh --cmd "$train_cmd" --nj 200 \
      data/$L/${train_set}_max2_hires $extractor exp/$L/nnet3/ivectors_${train_set}${ivector_suffix} || exit 1;
  fi

fi


exit 0;
