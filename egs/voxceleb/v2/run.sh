#!/bin/bash
# Copyright      2017   David Snyder
#                2017   Johns Hopkins University (Author: Daniel Garcia-Romero)
#                2017   Johns Hopkins University (Author: Daniel Povey)
#                2018   Ewald Enzinger
# Apache 2.0.
#
# Adapted from SRE16 v2 recipe (commit 3ea534070fd2cccd2e4ee21772132230033022ce)
#
# See ../README.txt for more info on data required.
# Results (mostly EERs) are inline in comments below.

. ./cmd.sh
. ./path.sh
set -e
mfccdir=`pwd`/mfcc
vaddir=`pwd`/mfcc


voxceleb1_trials=data/voxceleb1_test/trials
voxceleb1_root=/path/to/voxceleb1
voxceleb2_root=/path/to/voxceleb2
nnet_dir=exp/xvector_nnet_1a
musan_root=/export/corpora/JHU/musan

stage=0

. utils/parse_options.sh

if [ $stage -le 0 ]; then
  mkdir data
  local/make_voxceleb2.pl $voxceleb2_root dev data/voxceleb2_train
  local/make_voxceleb1_test.pl $voxceleb1_root data/voxceleb1_test
fi

if [ $stage -le 1 ]; then
  # Make MFCCs and compute the energy-based VAD for each dataset
  for name in voxceleb2_train voxceleb1_test; do
    steps/make_mfcc.sh --write-utt2num-frames true --mfcc-config conf/mfcc.conf --nj 40 --cmd "$train_cmd" \
      data/${name} exp/make_mfcc $mfccdir
    utils/fix_data_dir.sh data/${name}
    sid/compute_vad_decision.sh --nj 40 --cmd "$train_cmd" \
      data/${name} exp/make_vad $vaddir
    utils/fix_data_dir.sh data/${name}
  done
fi

# In this section, we augment the VoxCeleb2 data with reverberation,
# noise, music, and babble, and combine it with the clean data.
if [ $stage -le 2 ]; then
  frame_shift=0.01
  awk -v frame_shift=$frame_shift '{print $1, $2*frame_shift;}' data/voxceleb2_train/utt2num_frames > data/voxceleb2_train/reco2dur

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
  python steps/data/reverberate_data_dir.py \
    "${rvb_opts[@]}" \
    --speech-rvb-probability 1 \
    --pointsource-noise-addition-probability 0 \
    --isotropic-noise-addition-probability 0 \
    --num-replications 1 \
    --source-sampling-rate 16000 \
    data/voxceleb2_train data/voxceleb2_train_reverb
  cp data/voxceleb2_train/vad.scp data/voxceleb2_train_reverb/
  utils/copy_data_dir.sh --utt-suffix "-reverb" data/voxceleb2_train_reverb data/voxceleb2_train_reverb.new
  rm -rf data/voxceleb2_train_reverb
  mv data/voxceleb2_train_reverb.new data/voxceleb2_train_reverb

  # Prepare the MUSAN corpus, which consists of music, speech, and noise
  # suitable for augmentation.
  local/make_musan.sh $musan_root data

  # Get the duration of the MUSAN recordings.  This will be used by the
  # script augment_data_dir.py.
  for name in speech noise music; do
    utils/data/get_utt2dur.sh data/musan_${name}
    mv data/musan_${name}/utt2dur data/musan_${name}/reco2dur
  done

  # Augment with musan_noise
  python steps/data/augment_data_dir.py --utt-suffix "noise" --fg-interval 1 --fg-snrs "15:10:5:0" --fg-noise-dir "data/musan_noise" data/voxceleb2_train data/voxceleb2_train_noise
  # Augment with musan_music
  python steps/data/augment_data_dir.py --utt-suffix "music" --bg-snrs "15:10:8:5" --num-bg-noises "1" --bg-noise-dir "data/musan_music" data/voxceleb2_train data/voxceleb2_train_music
  # Augment with musan_speech
  python steps/data/augment_data_dir.py --utt-suffix "babble" --bg-snrs "20:17:15:13" --num-bg-noises "3:4:5:6:7" --bg-noise-dir "data/musan_speech" data/voxceleb2_train data/voxceleb2_train_babble

  # Combine reverb, noise, music, and babble into one directory.
  utils/combine_data.sh data/voxceleb2_train_aug data/voxceleb2_train_reverb data/voxceleb2_train_noise data/voxceleb2_train_music data/voxceleb2_train_babble
fi

if [ $stage -le 3 ]; then
  # Take a random subset of the augmentations
  utils/subset_data_dir.sh data/voxceleb2_train_aug 1000000 data/voxceleb2_train_aug_1m
  utils/fix_data_dir.sh data/voxceleb2_train_aug_1m

  # Make MFCCs for the augmented data.  Note that we do not compute a new
  # vad.scp file here.  Instead, we use the vad.scp from the clean version of
  # the list.
  steps/make_mfcc.sh --mfcc-config conf/mfcc.conf --nj 40 --cmd "$train_cmd" \
    data/voxceleb2_train_aug_1m exp/make_mfcc $mfccdir

  # Combine the clean and augmented VoxCeleb2 list.  This is now roughly
  # double the size of the original clean list.
  utils/combine_data.sh data/voxceleb2_train_combined data/voxceleb2_train_aug_1m data/voxceleb2_train
fi

# Now we prepare the features to generate examples for xvector training.
if [ $stage -le 4 ]; then
  # This script applies CMVN and removes nonspeech frames.  Note that this is somewhat
  # wasteful, as it roughly doubles the amount of training data on disk.  After
  # creating training examples, this can be removed.
  local/nnet3/xvector/prepare_feats_for_egs.sh --nj 40 --cmd "$train_cmd" \
    data/voxceleb2_train_combined data/voxceleb2_train_combined_no_sil exp/voxceleb2_train_combined_no_sil
  utils/fix_data_dir.sh data/voxceleb2_train_combined_no_sil
fi

if [ $stage -le 5 ]; then
  # Now, we need to remove features that are too short after removing silence
  # frames.  We want atleast 5s (500 frames) per utterance.
  min_len=500
  mv data/voxceleb2_train_combined_no_sil/utt2num_frames data/voxceleb2_train_combined_no_sil/utt2num_frames.bak
  awk -v min_len=${min_len} '$2 > min_len {print $1, $2}' data/voxceleb2_train_combined_no_sil/utt2num_frames.bak > data/voxceleb2_train_combined_no_sil/utt2num_frames
  utils/filter_scp.pl data/voxceleb2_train_combined_no_sil/utt2num_frames data/voxceleb2_train_combined_no_sil/utt2spk > data/voxceleb2_train_combined_no_sil/utt2spk.new
  mv data/voxceleb2_train_combined_no_sil/utt2spk.new data/voxceleb2_train_combined_no_sil/utt2spk
  utils/fix_data_dir.sh data/voxceleb2_train_combined_no_sil

  # We also want several utterances per speaker. Now we'll throw out speakers
  # with fewer than 8 utterances.
  min_num_utts=8
  awk '{print $1, NF-1}' data/voxceleb2_train_combined_no_sil/spk2utt > data/voxceleb2_train_combined_no_sil/spk2num
  awk -v min_num_utts=${min_num_utts} '$2 >= min_num_utts {print $1, $2}' data/voxceleb2_train_combined_no_sil/spk2num | utils/filter_scp.pl - data/voxceleb2_train_combined_no_sil/spk2utt > data/voxceleb2_train_combined_no_sil/spk2utt.new
  mv data/voxceleb2_train_combined_no_sil/spk2utt.new data/voxceleb2_train_combined_no_sil/spk2utt
  utils/spk2utt_to_utt2spk.pl data/voxceleb2_train_combined_no_sil/spk2utt > data/voxceleb2_train_combined_no_sil/utt2spk

  utils/filter_scp.pl data/voxceleb2_train_combined_no_sil/utt2spk data/voxceleb2_train_combined_no_sil/utt2num_frames > data/voxceleb2_train_combined_no_sil/utt2num_frames.new
  mv data/voxceleb2_train_combined_no_sil/utt2num_frames.new data/voxceleb2_train_combined_no_sil/utt2num_frames

  # Now we're ready to create training examples.
  utils/fix_data_dir.sh data/voxceleb2_train_combined_no_sil
fi

if [ $stage -le 6 ]; then
  local/nnet3/xvector/run_xvector.sh --stage $stage --train-stage -1 \
    --data data/voxceleb2_train_combined_no_sil --nnet-dir $nnet_dir \
    --egs-dir $nnet_dir/egs
fi

if [ $stage -le 7 ]; then
  sid/nnet3/xvector/extract_xvectors.sh --cmd "$train_cmd --mem 4G" --nj 80 \
    $nnet_dir data/voxceleb2_train \
    exp/xvectors_voxceleb2_train

  sid/nnet3/xvector/extract_xvectors.sh --cmd "$train_cmd --mem 4G" --nj 40 \
    $nnet_dir data/voxceleb1_test \
    exp/xvectors_voxceleb1_test
fi

if [ $stage -le 8 ]; then
  # Compute the mean vector for centering the evaluation xvectors.
  $train_cmd exp/xvectors_voxceleb2_train/log/compute_mean.log \
    ivector-mean scp:exp/xvectors_voxceleb2_train/xvector.scp \
    exp/xvectors_voxceleb2_train/mean.vec || exit 1;

  # This script uses LDA to decrease the dimensionality prior to PLDA.
  lda_dim=150
  $train_cmd exp/xvectors_voxceleb2_train/log/lda.log \
    ivector-compute-lda --total-covariance-factor=0.0 --dim=$lda_dim \
    "ark:ivector-subtract-global-mean scp:exp/xvectors_voxceleb2_train/xvector.scp ark:- |" \
    ark:data/voxceleb2_train/utt2spk exp/xvectors_voxceleb2_train/transform.mat || exit 1;

  # Train the PLDA model.
  $train_cmd exp/xvectors_voxceleb2_train/log/plda.log \
    ivector-compute-plda ark:data/voxceleb2_train/spk2utt \
    "ark:ivector-subtract-global-mean scp:exp/xvectors_voxceleb2_train/xvector.scp ark:- | transform-vec exp/xvectors_voxceleb2_train/transform.mat ark:- ark:- | ivector-normalize-length ark:-  ark:- |" \
    exp/xvectors_voxceleb2_train/plda || exit 1;
fi

if [ $stage -le 9 ]; then
  $train_cmd exp/scores/log/voxceleb1_test_scoring.log \
    ivector-plda-scoring --normalize-length=true \
    "ivector-copy-plda --smoothing=0.0 exp/xvectors_voxceleb2_train/plda - |" \
    "ark:ivector-subtract-global-mean exp/xvectors_voxceleb2_train/mean.vec scp:exp/xvectors_voxceleb1_test/xvector.scp ark:- | transform-vec exp/xvectors_voxceleb2_train/transform.mat ark:- ark:- | ivector-normalize-length ark:- ark:- |" \
    "ark:ivector-subtract-global-mean exp/xvectors_voxceleb2_train/mean.vec scp:exp/xvectors_voxceleb1_test/xvector.scp ark:- | transform-vec exp/xvectors_voxceleb2_train/transform.mat ark:- ark:- | ivector-normalize-length ark:- ark:- |" \
    "cat '$voxceleb1_trials' | cut -d\  --fields=1,2 |" exp/scores_voxceleb1_test || exit 1;
fi

if [ $stage -le 10 ]; then
  eer=`compute-eer <(local/prepare_for_eer.py $voxceleb1_trials exp/scores_voxceleb1_test) 2> /dev/null`
  echo "EER: ${eer}%"
  # EER: 4.173%
  # For reference, here's the ivector system from ../v1:
  # EER: 5.748%
fi
