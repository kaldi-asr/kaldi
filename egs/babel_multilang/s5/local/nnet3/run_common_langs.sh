#!/bin/bash

# Copyright 2016 Pegah Ghahremani

# This script used to generate MFCC+pitch features for input language lang.

. ./cmd.sh
set -e
stage=1
train_stage=-10
generate_alignments=true # If true, it regenerates alignments.
speed_perturb=true
use_pitch=true      # If true, it generates pitch features and combine it with 40dim MFCC.
pitch_conf=conf/pitch.conf # Configuration used for pitch extraction.
use_pitch_plp=false # If true, it generated plp+pitch to be used in regenerating alignments.

[ ! -f ./conf/common_vars.sh ] && echo 'the file conf/common_vars.sh does not exist!' && exit 1

. conf/common_vars.sh || exit 1;

[ -f local.conf ] && . ./local.conf

. ./utils/parse_options.sh

lang=$1

# perturbed data preparation
train_set=train
if [ "$speed_perturb" == "true" ]; then
  if [ $stage -le 1 ]; then
    #Although the nnet will be trained by high resolution data, we still have to perturbe the normal data to get the alignment
    # _sp stands for speed-perturbed
    for datadir in train; do
      if [ ! -d data/$lang/${datadir}_sp ]; then
        ./utils/data/perturb_data_dir_speed_3way.sh data/$lang/${datadir} data/$lang/${datadir}_sp

        # Extract Plp+pitch feature for perturbed data.
        featdir=plp_perturbed/$lang
        if $use_pitch_plp; then
          steps/make_plp_pitch.sh --cmd "$train_cmd" --nj $train_nj  data/$lang/${datadir}_sp exp/$lang/make_plp_pitch/${datadir}_sp $featdir
        else
          steps/make_plp.sh --cmd "$train_cmd" --nj $train_nj data/$lang/${datadir}_sp exp/$lang/make_plp/${datadir}_sp $featdir
        fi
        steps/compute_cmvn_stats.sh data/$lang/${datadir}_sp exp/$lang/make_plp/${datadir}_sp $featdir || exit 1;
        utils/fix_data_dir.sh data/$lang/${datadir}_sp
      fi
    done
  fi

  train_set=train_sp
  if [ $stage -le 2 ] && [ "$generate_alignments" == "true" ] && [ ! -f exp/$lang/tri5_ali_sp/.done ]; then
    #obtain the alignment of the perturbed data
    steps/align_fmllr.sh \
      --nj 70 --cmd "$train_cmd" \
      --boost-silence $boost_sil \
      data/$lang/$train_set data/$lang/lang exp/$lang/tri5 exp/$lang/tri5_ali_sp || exit 1
    touch exp/$lang/tri5_ali_sp/.done
  fi
fi

if [ $stage -le 3 ] && [ ! -f data/$lang/${train_set}_hires/.done ]; then
  mfccdir=mfcc_hires/$lang
  if [[ $(hostname -f) == *.clsp.jhu.edu ]] && [ ! -d $mfccdir/storage ]; then
    date=$(date +'%m_%d_%H_%M')
    utils/create_split_dir.pl /export/b0{1,2,3,4}/$USER/kaldi-data/egs/$lang-$date/s5c/$mfccdir/storage $mfccdir/storage
  fi

  for dataset in $train_set ; do
    utils/copy_data_dir.sh data/$lang/$dataset data/$lang/${dataset}_hires

    # scale the waveforms, this is useful as we don't use CMVN
    data_dir=data/$lang/${dataset}_hires

    utils/data/perturb_data_dir_volume.sh $data_dir || exit 1 ;

    steps/make_mfcc.sh --nj 70 --mfcc-config conf/mfcc_hires.conf \
      --cmd "$train_cmd" data/$lang/${dataset}_hires exp/$lang/make_hires/$dataset $mfccdir;

    steps/compute_cmvn_stats.sh data/$lang/${dataset}_hires exp/$lang/make_hires/${dataset} $mfccdir;

    # Remove the small number of utterances that couldn't be extracted for some
    # reason (e.g. too short; no such file).
    utils/fix_data_dir.sh data/$lang/${dataset}_hires;
  done
  touch data/$lang/${train_set}_hires/.done
fi

if [ $stage -le 4 ]; then
  if [[ "$use_pitch" == "true" ]]; then
    pitchdir=pitch/$lang
    train_set=${train_set}_hires
    for dataset in $train_set; do
      if $use_pitch; then
        mkdir -p $pitchdir
        if [ ! -f data/$lang/${dataset}_pitch/feats.scp ]; then
          echo "$0: Generating pitch features for data/$lang as use_pitch=$use_pitch"
          utils/copy_data_dir.sh data/$lang/$dataset data/$lang/${dataset}_pitch
          steps/make_pitch.sh --nj 70 --pitch-config $pitch_conf \
            --cmd "$train_cmd" data/$lang/${dataset}_pitch exp/$lang/make_pitch/${dataset} $pitchdir;
        fi
        feat_suffix=_pitch
      fi

      if [ ! -f data/$lang/${dataset}_mfcc${feat_suffix}/feats.scp ]; then
        steps/append_feats.sh --nj 16 --cmd "$train_cmd" data/$lang/${dataset} \
          data/$lang/${dataset}${feat_suffix} data/$lang/${dataset}_mfcc${feat_suffix} \
          exp/$lang/append_mfcc${feat_suffix}/${dataset} mfcc${feat_suffix}/$lang

        steps/compute_cmvn_stats.sh data/$lang/${dataset}_mfcc${feat_suffix} exp/$lang/make_cmvn_mfcc${feat_suffix}/${x} mfcc${feat_suffix}/$lang
      fi
    done
  fi
fi

exit 0;
