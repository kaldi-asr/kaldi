#!/usr/bin/env bash
# Copyright   2017   Johns Hopkins University (Author: Daniel Garcia-Romero)
#             2017   Johns Hopkins University (Author: Daniel Povey)
#        2017-2018   David Snyder
#             2018   Ewald Enzinger
# Apache 2.0.
#
# This is an i-vector-based recipe for Speakers in the Wild (SITW).
# See ../README.txt for more info on data required.  The recipe uses
# VoxCeleb 1 and 2 for training the UBM and T matrix, and an augmented
# version of those datasets for PLDA training.  The augmentation consists
# of MUSAN noises, music, and babble and reverberation from the Room
# Impulse Response and Noise Database.  Note that there are 60 speakers
# in VoxCeleb 1 that overlap with our evaluation dataset, SITW.  The recipe
# removes those 60 speakers prior to training.  See ../README.txt for more
# info on data required.  The results are reported in terms of EER and minDCF,
# and are inline in the comments below.

. ./cmd.sh
. ./path.sh
set -e
mfccdir=`pwd`/mfcc
vaddir=`pwd`/mfcc

voxceleb1_root=/export/corpora/VoxCeleb1
voxceleb2_root=/export/corpora/VoxCeleb2
sitw_root=/export/corpora/SRI/sitw
musan_root=/export/corpora/JHU/musan

sitw_dev_trials_core=data/sitw_dev_test/trials/core-core.lst
sitw_eval_trials_core=data/sitw_eval_test/trials/core-core.lst

stage=0

if [ $stage -le 0 ]; then
  # Prepare the VoxCeleb1 dataset.  The script also downloads a list from
  # http://www.openslr.org/resources/49/voxceleb1_sitw_overlap.txt that
  # contains the speakers that overlap between VoxCeleb1 and our evaluation
  # set SITW.  The script removes the overlapping speakers from VoxCeleb1.
  local/make_voxceleb1.pl $voxceleb1_root data

  # Prepare the dev portion of the VoxCeleb2 dataset.
  local/make_voxceleb2.pl $voxceleb2_root dev data/voxceleb2_train

  # The original version of this recipe included the test portion of VoxCeleb2
  # in the training list.  Unfortunately, it turns out that there's an overlap
  # with our evaluation set, Speakers in the Wild.  Therefore, we've removed
  # this dataset from the training list.
  # local/make_voxceleb2.pl $voxceleb2_root test data/voxceleb2_test

  # We'll train on all of VoxCeleb2, plus the training portion of VoxCeleb1.
  # This should leave 7,185 speakers and 1,236,567 utterances.
  utils/combine_data.sh data/train data/voxceleb2_train data/voxceleb1

  # Prepare Speakers in the Wild.  This is our evaluation dataset.
  local/make_sitw.sh $sitw_root data
fi

if [ $stage -le 1 ]; then
  # Make MFCCs and compute the energy-based VAD for each dataset
  for name in sitw_eval_enroll sitw_eval_test sitw_dev_enroll sitw_dev_test train; do
    steps/make_mfcc.sh --write-utt2num-frames true --mfcc-config conf/mfcc.conf --nj 80 --cmd "$train_cmd" \
      data/${name} exp/make_mfcc $mfccdir
    utils/fix_data_dir.sh data/${name}
    sid/compute_vad_decision.sh --nj 80 --cmd "$train_cmd" \
      data/${name} exp/make_vad $vaddir
    utils/fix_data_dir.sh data/${name}
  done
fi

if [ $stage -le 2 ]; then
  # Train the UBM on VoxCeleb 1 and 2.
  sid/train_diag_ubm.sh --cmd "$train_cmd --mem 4G" \
    --nj 40 --num-threads 8 \
    data/train 2048 \
    exp/diag_ubm

  sid/train_full_ubm.sh --cmd "$train_cmd --mem 25G" \
    --nj 40 --remove-low-count-gaussians false \
    data/train \
    exp/diag_ubm exp/full_ubm
fi

if [ $stage -le 3 ]; then
  # In this stage, we train the i-vector extractor on a subset of VoxCeleb 1
  # and 2.
  #
  # Note that there are well over 1 million utterances in our training set,
  # and it takes an extremely long time to train the extractor on all of this.
  # Also, most of those utterances are very short.  Short utterances are
  # harmful for training the i-vector extractor.  Therefore, to reduce the
  # training time and improve performance, we will only train on the 100k
  # longest utterances.
  utils/subset_data_dir.sh \
    --utt-list <(sort -n -k 2 data/train/utt2num_frames | tail -n 100000) \
    data/train data/train_100k

  # Train the i-vector extractor.
  sid/train_ivector_extractor.sh --cmd "$train_cmd --mem 16G" \
    --ivector-dim 400 --num-iters 5 \
    exp/full_ubm/final.ubm data/train_100k \
    exp/extractor
fi

# In this section, we augment the VoxCeleb 1 and 2 data with reverberation,
# noise, music, and babble, and combine it with the clean data.  This will
# later be used to train out PLDA model.
if [ $stage -le 4 ]; then
  frame_shift=0.01
  awk -v frame_shift=$frame_shift '{print $1, $2*frame_shift;}' data/train_100k/utt2num_frames > data/train_100k/reco2dur

  if [ ! -d "RIRS_NOISES" ]; then
    # Download the package that includes the real RIRs, simulated RIRs, isotropic noises and point-source noises
    wget --no-check-certificate http://www.openslr.org/resources/28/rirs_noises.zip
    unzip rirs_noises.zip
  fi

  # Make a version with reverberated speech
  rvb_opts=()
  rvb_opts+=(--rir-set-parameters "0.5, RIRS_NOISES/simulated_rirs/smallroom/rir_list")
  rvb_opts+=(--rir-set-parameters "0.5, RIRS_NOISES/simulated_rirs/mediumroom/rir_list")

  # Make a reverberated version of the VoxCeleb2 list.  Note that we don't add any
  # additive noise here.
  steps/data/reverberate_data_dir.py \
    "${rvb_opts[@]}" \
    --speech-rvb-probability 1 \
    --pointsource-noise-addition-probability 0 \
    --isotropic-noise-addition-probability 0 \
    --num-replications 1 \
    --source-sampling-rate 16000 \
    data/train_100k data/train_100k_reverb
  cp data/train_100k/vad.scp data/train_100k_reverb/
  utils/copy_data_dir.sh --utt-suffix "-reverb" data/train_100k_reverb data/train_100k_reverb.new
  rm -rf data/train_100k_reverb
  mv data/train_100k_reverb.new data/train_100k_reverb

  # Prepare the MUSAN corpus, which consists of music, speech, and noise
  # suitable for augmentation.
  steps/data/make_musan.sh --sampling-rate 16000 $musan_root data

  # Get the duration of the MUSAN recordings.  This will be used by the
  # script augment_data_dir.py.
  for name in speech noise music; do
    utils/data/get_utt2dur.sh data/musan_${name}
    mv data/musan_${name}/utt2dur data/musan_${name}/reco2dur
  done

  # Augment with musan_noise
  steps/data/augment_data_dir.py --utt-suffix "noise" --fg-interval 1 --fg-snrs "15:10:5:0" --fg-noise-dir "data/musan_noise" data/train_100k data/train_100k_noise
  # Augment with musan_music
  steps/data/augment_data_dir.py --utt-suffix "music" --bg-snrs "15:10:8:5" --num-bg-noises "1" --bg-noise-dir "data/musan_music" data/train_100k data/train_100k_music
  # Augment with musan_speech
  steps/data/augment_data_dir.py --utt-suffix "babble" --bg-snrs "20:17:15:13" --num-bg-noises "3:4:5:6:7" --bg-noise-dir "data/musan_speech" data/train_100k data/train_100k_babble

  # Combine reverb, noise, music, and babble into one directory.
  utils/combine_data.sh data/train_aug data/train_100k_reverb data/train_100k_noise data/train_100k_music data/train_100k_babble
fi

if [ $stage -le 5 ]; then
  # Take a 100k subset of the augmentations.
  utils/subset_data_dir.sh data/train_aug 100000 data/train_100k_aug
  utils/fix_data_dir.sh data/train_100k_aug

  # Make MFCCs for the augmented data.  Note that we do not compute a new
  # vad.scp file here.  Instead, we use the vad.scp from the clean version of
  # the list.
  steps/make_mfcc.sh --mfcc-config conf/mfcc.conf --nj 80 --cmd "$train_cmd" \
    data/train_100k_aug exp/make_mfcc $mfccdir

  # Combine the clean and augmented VoxCeleb list.  This is now roughly
  # double the size of the original clean list.
  utils/combine_data.sh data/train_combined_200k data/train_100k_aug data/train_100k
fi

if [ $stage -le 6 ]; then
  # These i-vectors will be used for mean-subtraction, LDA, and PLDA training.
  sid/extract_ivectors.sh --cmd "$train_cmd --mem 4G" --nj 80 \
    exp/extractor data/train_combined_200k \
    exp/ivectors_train_combined_200k

  # Extract i-vectors for the SITW dev and eval sets.
  for name in sitw_eval_enroll sitw_eval_test sitw_dev_enroll sitw_dev_test; do
    sid/extract_ivectors.sh --cmd "$train_cmd --mem 4G" --nj 40 \
      exp/extractor data/$name \
      exp/ivectors_$name
  done
fi

if [ $stage -le 7 ]; then
  # Compute the mean vector for centering the evaluation i-vectors.
  $train_cmd exp/ivectors_train_combined_200k/log/compute_mean.log \
    ivector-mean scp:exp/ivectors_train_combined_200k/ivector.scp \
    exp/ivectors_train_combined_200k/mean.vec || exit 1;

  # This script uses LDA to decrease the dimensionality prior to PLDA.
  lda_dim=150
  $train_cmd exp/ivectors_train_combined_200k/log/lda.log \
    ivector-compute-lda --total-covariance-factor=0.0 --dim=$lda_dim \
    "ark:ivector-subtract-global-mean scp:exp/ivectors_train_combined_200k/ivector.scp ark:- |" \
    ark:data/train_combined_200k/utt2spk exp/ivectors_train_combined_200k/transform.mat || exit 1;

  # Train the PLDA model.
  $train_cmd exp/ivectors_train_combined_200k/log/plda.log \
    ivector-compute-plda ark:data/train_combined_200k/spk2utt \
    "ark:ivector-subtract-global-mean scp:exp/ivectors_train_combined_200k/ivector.scp ark:- | transform-vec exp/ivectors_train_combined_200k/transform.mat ark:- ark:- | ivector-normalize-length ark:-  ark:- |" \
    exp/ivectors_train_combined_200k/plda || exit 1;
fi

if [ $stage -le 8 ]; then
  # Compute PLDA scores for SITW dev core-core trials
  $train_cmd exp/scores/log/sitw_dev_core_scoring.log \
    ivector-plda-scoring --normalize-length=true \
    --num-utts=ark:exp/ivectors_sitw_dev_enroll/num_utts.ark \
    "ivector-copy-plda --smoothing=0.0 exp/ivectors_train_combined_200k/plda - |" \
    "ark:ivector-mean ark:data/sitw_dev_enroll/spk2utt scp:exp/ivectors_sitw_dev_enroll/ivector.scp ark:- | ivector-subtract-global-mean exp/ivectors_train_combined_200k/mean.vec ark:- ark:- | transform-vec exp/ivectors_train_combined_200k/transform.mat ark:- ark:- | ivector-normalize-length ark:- ark:- |" \
    "ark:ivector-subtract-global-mean exp/ivectors_train_combined_200k/mean.vec scp:exp/ivectors_sitw_dev_test/ivector.scp ark:- | transform-vec exp/ivectors_train_combined_200k/transform.mat ark:- ark:- | ivector-normalize-length ark:- ark:- |" \
    "cat '$sitw_dev_trials_core' | cut -d\  --fields=1,2 |" exp/scores/sitw_dev_core_scores || exit 1;

  # SITW Dev Core:
  # EER: 4.813%
  # minDCF(p-target=0.01): 0.4250
  # minDCF(p-target=0.001): 0.5727
  echo "SITW Dev Core:"
  eer=$(paste $sitw_dev_trials_core exp/scores/sitw_dev_core_scores | awk '{print $6, $3}' | compute-eer - 2>/dev/null)
  mindcf1=`sid/compute_min_dcf.py --p-target 0.01 exp/scores/sitw_dev_core_scores $sitw_dev_trials_core 2> /dev/null`
  mindcf2=`sid/compute_min_dcf.py --p-target 0.001 exp/scores/sitw_dev_core_scores $sitw_dev_trials_core 2> /dev/null`
  echo "EER: $eer%"
  echo "minDCF(p-target=0.01): $mindcf1"
  echo "minDCF(p-target=0.001): $mindcf2"
fi

if [ $stage -le 9 ]; then
  # Compute PLDA scores for SITW eval core-core trials
  $train_cmd exp/scores/log/sitw_eval_core_scoring.log \
    ivector-plda-scoring --normalize-length=true \
    --num-utts=ark:exp/ivectors_sitw_eval_enroll/num_utts.ark \
    "ivector-copy-plda --smoothing=0.0 exp/ivectors_train_combined_200k/plda - |" \
    "ark:ivector-mean ark:data/sitw_eval_enroll/spk2utt scp:exp/ivectors_sitw_eval_enroll/ivector.scp ark:- | ivector-subtract-global-mean exp/ivectors_train_combined_200k/mean.vec ark:- ark:- | transform-vec exp/ivectors_train_combined_200k/transform.mat ark:- ark:- | ivector-normalize-length ark:- ark:- |" \
    "ark:ivector-subtract-global-mean exp/ivectors_train_combined_200k/mean.vec scp:exp/ivectors_sitw_eval_test/ivector.scp ark:- | transform-vec exp/ivectors_train_combined_200k/transform.mat ark:- ark:- | ivector-normalize-length ark:- ark:- |" \
    "cat '$sitw_eval_trials_core' | cut -d\  --fields=1,2 |" exp/scores/sitw_eval_core_scores || exit 1;

  # SITW Eval Core:
  # EER: 5.659%
  # minDCF(p-target=0.01): 0.4637
  # minDCF(p-target=0.001): 0.6290
  echo -e "\nSITW Eval Core:";
  eer=$(paste $sitw_eval_trials_core exp/scores/sitw_eval_core_scores | awk '{print $6, $3}' | compute-eer - 2>/dev/null)
  mindcf1=`sid/compute_min_dcf.py --p-target 0.01 exp/scores/sitw_eval_core_scores $sitw_eval_trials_core 2> /dev/null`
  mindcf2=`sid/compute_min_dcf.py --p-target 0.001 exp/scores/sitw_eval_core_scores $sitw_eval_trials_core 2> /dev/null`
  echo "EER: $eer%"
  echo "minDCF(p-target=0.01): $mindcf1"
  echo "minDCF(p-target=0.001): $mindcf2"
fi
