#!/bin/bash

set -euo pipefail

# This script is called from local/nnet3/run_tdnn.sh and
# local/chain/run_tdnn.sh (and may eventually be called by more
# scripts).  It contains the common feature preparation and
# iVector-related parts of the script.  See those scripts for examples
# of usage.

stage=0
train_set_clean=train_worn
train_set_noisy=train_u400k
combined_train_set=train_worn_u400k
test_sets="dev_worn"
nj=96
include_clean=false

noise_list=
num_data_reps=2
snrs="20:10:15:5:0"
foreground_snrs="20:10:15:5:0"
background_snrs="20:10:15:5:0"

nnet3_affix=_train_worn_u400k_rvb

. ./cmd.sh
. ./path.sh
. utils/parse_options.sh

if [ $stage -le 0 ]; then
  # Perturb the original data. We need this Although the nnet will be trained by high resolution data, we still have to
  # perturb the normal data to get the alignment _sp stands for speed-perturbed
  echo "$0: preparing directory for low-resolution speed-perturbed data (for alignment)"
  utils/data/perturb_data_dir_speed_3way.sh data/${train_set_clean} data/${train_set_clean}_sp

  utils/data/perturb_data_dir_speed_3way.sh data/${train_set_noisy} data/${train_set_noisy}_sp

  for datadir in ${train_set_clean}_sp ${train_set_noisy}_sp $test_sets; do
    utils/copy_data_dir.sh data/$datadir data/${datadir}_hires
  done
  
  for datadir in ${train_set_clean}_sp ${train_set_noisy}_sp; do
    utils/data/perturb_data_dir_volume.sh data/${datadir}_hires
  done
fi
 

if [ $stage -le 1 ]; then
  for datadir in ${train_set_clean}_sp ${train_set_noisy}_sp; do
    mfccdir=data/${datadir}/data
    if [[ $(hostname -f) == *.clsp.jhu.edu ]] && [ ! -d $mfccdir/storage ]; then
      utils/create_split_dir.pl /export/b0{5,6,7,8}/$USER/kaldi-data/mfcc/chime5-$(date +'%m_%d_%H_%M')/s5/$mfccdir/storage $mfccdir/storage
    fi

    steps/make_mfcc.sh --nj 20 \
      --cmd "$train_cmd" data/${datadir} || exit 1;
    steps/compute_cmvn_stats.sh data/${datadir} || exit 1;
    utils/fix_data_dir.sh data/${datadir} || exit 1;
  done
fi

if [ $stage -le 2 ]; then
  if [ ! -d RIRS_NOISES/ ]; then
    # Download the package that includes the real RIRs, simulated RIRs, isotropic noises and point-source noises
    wget --no-check-certificate http://www.openslr.org/resources/28/rirs_noises.zip
    unzip rirs_noises.zip
  fi

  if [ -z "$noise_list" ]; then
    noise_list=RIRS_NOISES/pointsource_noises/noise_list
  fi

  # This is the config for the system using simulated RIRs and point-source noises
  rvb_opts+=(--rir-set-parameters "0.5, RIRS_NOISES/simulated_rirs/smallroom/rir_list")
  rvb_opts+=(--rir-set-parameters "0.5, RIRS_NOISES/simulated_rirs/mediumroom/rir_list")
  rvb_opts+=(--noise-set-parameters $noise_list)

  python steps/data/reverberate_data_dir.py \
    "${rvb_opts[@]}" \
    --prefix "rev" \
    --foreground-snrs $foreground_snrs \
    --background-snrs $background_snrs \
    --speech-rvb-probability 1 \
    --pointsource-noise-addition-probability 1 \
    --isotropic-noise-addition-probability 1 \
    --num-replications $num_data_reps \
    --max-noises-per-minute 1 \
    --source-sampling-rate 16000 \
    data/${train_set_clean}_sp_hires data/${train_set_clean}_sp_rvb_hires
fi

if [ $stage -le 3 ]; then
  # Create high-resolution MFCC features (with 40 cepstra instead of 13).
  # this shows how you can split across multiple file-systems.
  echo "$0: creating high-resolution MFCC features"
  for datadir in ${train_set_clean}_sp_rvb ${train_set_noisy}_sp ${test_sets}; do
    mfccdir=data/${datadir}_hires/data
    if [[ $(hostname -f) == *.clsp.jhu.edu ]] && [ ! -d $mfccdir/storage ]; then
      utils/create_split_dir.pl /export/b0{5,6,7,8}/$USER/kaldi-data/mfcc/chime5-$(date +'%m_%d_%H_%M')/s5/$mfccdir/storage $mfccdir/storage
    fi
  
    steps/make_mfcc.sh --nj 20 --mfcc-config conf/mfcc_hires.conf \
      --cmd "$train_cmd" data/${datadir}_hires || exit 1;
    steps/compute_cmvn_stats.sh data/${datadir}_hires || exit 1;
    utils/fix_data_dir.sh data/${datadir}_hires || exit 1;
  done
fi
  
temp_data_root=exp/nnet3${nnet3_affix}/diag_ubm

if [ $stage -le 4 ]; then
  echo "$0: computing a subset of data to train the diagonal UBM."
  # We'll use about a quarter of the data.
  mkdir -p exp/nnet3${nnet3_affix}/diag_ubm
  optional_clean=
  if $include_clean; then
    optional_clean=data/${train_set_clean}_sp_hires
  fi
  utils/combine_data.sh data/${combined_train_set}_sp_rvb_hires \
    ${optional_clean} \
    data/${train_set_clean}_sp_rvb_hires data/${train_set_noisy}_sp_hires

  num_utts_total=$(wc -l < data/${combined_train_set}_sp_rvb_hires/utt2spk)
  num_utts=$[$num_utts_total/4]
  utils/data/subset_data_dir.sh data/${combined_train_set}_sp_rvb_hires \
     $num_utts ${temp_data_root}/${combined_train_set}_sp_rvb_hires_subset

  echo "$0: computing a PCA transform from the hires data."
  steps/online/nnet2/get_pca_transform.sh --cmd "$train_cmd" \
      --splice-opts "--left-context=3 --right-context=3" \
      --max-utts 10000 --subsample 2 \
      ${temp_data_root}/${combined_train_set}_sp_rvb_hires_subset \
      exp/nnet3${nnet3_affix}/pca_transform

  echo "$0: training the diagonal UBM."
  # Use 512 Gaussians in the UBM.
  steps/online/nnet2/train_diag_ubm.sh --cmd "$train_cmd" --nj 30 \
    --num-frames 700000 \
    --num-threads 8 \
    data/${combined_train_set}_sp_rvb_hires 512 \
    exp/nnet3${nnet3_affix}/pca_transform exp/nnet3${nnet3_affix}/diag_ubm
fi

if [ $stage -le 5 ]; then
  # Train the iVector extractor.  Use all of the speed-perturbed data since iVector extractors
  # can be sensitive to the amount of data.  The script defaults to an iVector dimension of
  # 100.
  echo "$0: training the iVector extractor"
  steps/online/nnet2/train_ivector_extractor.sh --cmd "$train_cmd" --nj 20 \
    data/${combined_train_set}_sp_rvb_hires exp/nnet3${nnet3_affix}/diag_ubm \
    exp/nnet3${nnet3_affix}/extractor || exit 1;
fi


if [ $stage -le 6 ]; then
  # We extract iVectors on the speed-perturbed training data after combining
  # short segments, which will be what we train the system on.  With
  # --utts-per-spk-max 2, the script pairs the utterances into twos, and treats
  # each of these pairs as one speaker; this gives more diversity in iVectors..
  # Note that these are extracted 'online'.

  # note, we don't encode the 'max2' in the name of the ivectordir even though
  # that's the data we extract the ivectors from, as it's still going to be
  # valid for the non-'max2' data, the utterance list is the same.

  for datadir in ${combined_train_set}_sp_rvb; do
    ivectordir=exp/nnet3${nnet3_affix}/ivectors_${datadir}_hires
    if [[ $(hostname -f) == *.clsp.jhu.edu ]] && [ ! -d $ivectordir/storage ]; then
      utils/create_split_dir.pl /export/b0{5,6,7,8}/$USER/kaldi-data/ivectors/chime5-$(date +'%m_%d_%H_%M')/s5/$ivectordir/storage $ivectordir/storage
    fi

    # having a larger number of speakers is helpful for generalization, and to
    # handle per-utterance decoding well (iVector starts at zero).
    temp_data_root=${ivectordir}
    utils/data/modify_speaker_info.sh --utts-per-spk-max 2 \
      data/${datadir}_hires ${temp_data_root}/${datadir}_hires_max2

    steps/online/nnet2/extract_ivectors_online.sh --cmd "$train_cmd" --nj ${nj} \
      ${temp_data_root}/${datadir}_hires_max2 \
      exp/nnet3${nnet3_affix}/extractor $ivectordir
  done

  # Also extract iVectors for the test data, but in this case we don't need the speed
  # perturbation (sp).
  for data in $test_sets; do
    steps/online/nnet2/extract_ivectors_online.sh --cmd "$train_cmd" --nj 20 \
      data/${data}_hires exp/nnet3${nnet3_affix}/extractor \
      exp/nnet3${nnet3_affix}/ivectors_${data}_hires
  done
fi

exit 0

