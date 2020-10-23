#!/usr/bin/env bash
# Copyright    2017   Johns Hopkins University (Author: Daniel Povey)
#              2017   Johns Hopkins University (Author: Daniel Garcia-Romero)
#              2018   Ewald Enzinger
#              2018   David Snyder
# Apache 2.0.
#
# This is an x-vector-based recipe for Speakers in the Wild (SITW).
# It is based on "X-vectors: Robust DNN Embeddings for Speaker Recognition"
# by Snyder et al.  The recipe uses augmented VoxCeleb 1 and 2 for training.
# The augmentation consists of MUSAN noises, music, and babble and
# reverberation from the Room Impulse Response and Noise Database.  Note that
# there are 60 speakers in VoxCeleb 1 that overlap with our evaluation
# dataset, SITW.  The recipe removes those 60 speakers prior to training.
# See ../README.txt for more info on data required.  The results are reported
# in terms of EER and minDCF, and are inline in the comments below.

. ./cmd.sh
. ./path.sh
set -e
mfccdir=`pwd`/mfcc
vaddir=`pwd`/mfcc

voxceleb1_root=/export/corpora/VoxCeleb1
voxceleb2_root=/export/corpora/VoxCeleb2
sitw_root=/export/corpora/SRI/sitw
nnet_dir=exp/xvector_nnet_1a
musan_root=/export/corpora/JHU/musan

sitw_dev_trials_core=data/sitw_dev_test/trials/core-core.lst
sitw_eval_trials_core=data/sitw_eval_test/trials/core-core.lst

stage=0

if [ $stage -le 0 ]; then
  # Prepare the VoxCeleb1 dataset.  The script also downloads a list from
  # http://www.openslr.org/resources/49/voxceleb1_sitw_overlap.txt that
  # contains the speakers that overlap between VoxCeleb1 and our evaluation
  # set SITW.  The script removes these overlapping speakers from VoxCeleb1.
  #local/make_voxceleb1.pl $voxceleb1_root data
  local/make_voxceleb1_v2.pl $voxceleb1_root data
  # Prepare the dev portion of the VoxCeleb2 dataset.
  local/make_voxceleb2.pl $voxceleb2_root dev data/voxceleb2_train

  # The original version of this recipe included the test portion of VoxCeleb2
  # in the training list.  Unfortunately, it turns out that there's an overlap
  # with our evaluation set, Speakers in the Wild.  Therefore, we've removed
  # this dataset from the training list.
  # local/make_voxceleb2.pl $voxceleb2_root test data/voxceleb2_test

  # We'll train on the dev portion of VoxCeleb2, plus VoxCeleb1 (minus the
  # speakers that overlap with SITW).
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

# In this section, we augment the VoxCeleb2 data with reverberation,
# noise, music, and babble, and combine it with the clean data.
if [ $stage -le 2 ]; then
  frame_shift=0.01
  awk -v frame_shift=$frame_shift '{print $1, $2*frame_shift;}' data/train/utt2num_frames > data/train/reco2dur

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
    data/train data/train_reverb
  cp data/train/vad.scp data/train_reverb/
  utils/copy_data_dir.sh --utt-suffix "-reverb" data/train_reverb data/train_reverb.new
  rm -rf data/train_reverb
  mv data/train_reverb.new data/train_reverb

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
  steps/data/augment_data_dir.py --utt-suffix "noise" --fg-interval 1 --fg-snrs "15:10:5:0" --fg-noise-dir "data/musan_noise" data/train data/train_noise
  # Augment with musan_music
  steps/data/augment_data_dir.py --utt-suffix "music" --bg-snrs "15:10:8:5" --num-bg-noises "1" --bg-noise-dir "data/musan_music" data/train data/train_music
  # Augment with musan_speech
  steps/data/augment_data_dir.py --utt-suffix "babble" --bg-snrs "20:17:15:13" --num-bg-noises "3:4:5:6:7" --bg-noise-dir "data/musan_speech" data/train data/train_babble

  # Combine reverb, noise, music, and babble into one directory.
  utils/combine_data.sh data/train_aug data/train_reverb data/train_noise data/train_music data/train_babble
fi

if [ $stage -le 3 ]; then
  # Take a random subset of the augmentations
  utils/subset_data_dir.sh data/train_aug 1000000 data/train_aug_1m
  utils/fix_data_dir.sh data/train_aug_1m

  # Make MFCCs for the augmented data.  Note that we do not compute a new
  # vad.scp file here.  Instead, we use the vad.scp from the clean version of
  # the list.
  steps/make_mfcc.sh --mfcc-config conf/mfcc.conf --nj 80 --cmd "$train_cmd" \
    data/train_aug_1m exp/make_mfcc $mfccdir

  # Combine the clean and augmented VoxCeleb2 list.  This is now roughly
  # double the size of the original clean list.
  utils/combine_data.sh data/train_combined data/train_aug_1m data/train
fi

# Now we prepare the features to generate examples for xvector training.
if [ $stage -le 4 ]; then
  # This script applies CMVN and removes nonspeech frames.  Note that this is somewhat
  # wasteful, as it roughly doubles the amount of training data on disk.  After
  # creating training examples, this can be removed.
  local/nnet3/xvector/prepare_feats_for_egs.sh --nj 80 --cmd "$train_cmd" \
    data/train_combined data/train_combined_no_sil exp/train_combined_no_sil
  utils/fix_data_dir.sh data/train_combined_no_sil
fi

if [ $stage -le 5 ]; then
  # Now, we need to remove features that are too short after removing silence
  # frames.  We want atleast 5s (500 frames) per utterance.
  min_len=400
  mv data/train_combined_no_sil/utt2num_frames data/train_combined_no_sil/utt2num_frames.bak
  awk -v min_len=${min_len} '$2 > min_len {print $1, $2}' data/train_combined_no_sil/utt2num_frames.bak > data/train_combined_no_sil/utt2num_frames
  utils/filter_scp.pl data/train_combined_no_sil/utt2num_frames data/train_combined_no_sil/utt2spk > data/train_combined_no_sil/utt2spk.new
  mv data/train_combined_no_sil/utt2spk.new data/train_combined_no_sil/utt2spk
  utils/fix_data_dir.sh data/train_combined_no_sil

  # We also want several utterances per speaker. Now we'll throw out speakers
  # with fewer than 8 utterances.
  min_num_utts=8
  awk '{print $1, NF-1}' data/train_combined_no_sil/spk2utt > data/train_combined_no_sil/spk2num
  awk -v min_num_utts=${min_num_utts} '$2 >= min_num_utts {print $1, $2}' data/train_combined_no_sil/spk2num | utils/filter_scp.pl - data/train_combined_no_sil/spk2utt > data/train_combined_no_sil/spk2utt.new
  mv data/train_combined_no_sil/spk2utt.new data/train_combined_no_sil/spk2utt
  utils/spk2utt_to_utt2spk.pl data/train_combined_no_sil/spk2utt > data/train_combined_no_sil/utt2spk

  utils/filter_scp.pl data/train_combined_no_sil/utt2spk data/train_combined_no_sil/utt2num_frames > data/train_combined_no_sil/utt2num_frames.new
  mv data/train_combined_no_sil/utt2num_frames.new data/train_combined_no_sil/utt2num_frames

  # Now we're ready to create training examples.
  utils/fix_data_dir.sh data/train_combined_no_sil
fi

# Stages 6 through 8 are handled in run_xvector.sh
local/nnet3/xvector/run_xvector.sh --stage $stage --train-stage -1 \
  --data data/train_combined_no_sil --nnet-dir $nnet_dir \
  --egs-dir $nnet_dir/egs

if [ $stage -le 9 ]; then
   # Now we will extract x-vectors used for centering, LDA, and PLDA training.
   # Note that data/train_combined has well over 2 million utterances,
   # which is far more than is needed to train the generative PLDA model.
   # In addition, many of the utterances are very short, which causes a
   # mismatch with our evaluation conditions.  In the next command, we
   # create a data directory that contains the longest 200,000 recordings,
   # which we will use to train the backend.
   utils/subset_data_dir.sh \
     --utt-list <(sort -n -k 2 data/train_combined_no_sil/utt2num_frames | tail -n 200000) \
     data/train_combined data/train_combined_200k

   sid/nnet3/xvector/extract_xvectors.sh --cmd "$train_cmd --mem 4G" --nj 80 \
    $nnet_dir data/train_combined_200k \
    $nnet_dir/xvectors_train_combined_200k

  # Extract x-vectors used in the evaluation.
  for name in sitw_eval_enroll sitw_eval_test sitw_dev_enroll sitw_dev_test; do
    sid/nnet3/xvector/extract_xvectors.sh --cmd "$train_cmd --mem 4G" --nj 40 \
      $nnet_dir data/$name \
      $nnet_dir/xvectors_$name
  done
fi

if [ $stage -le 10 ]; then
  # Compute the mean.vec used for centering.
  $train_cmd $nnet_dir/xvectors_train_combined_200k/log/compute_mean.log \
    ivector-mean scp:$nnet_dir/xvectors_train_combined_200k/xvector.scp \
    $nnet_dir/xvectors_train_combined_200k/mean.vec || exit 1;

  # Use LDA to decrease the dimensionality prior to PLDA.
  lda_dim=128
  $train_cmd $nnet_dir/xvectors_train_combined_200k/log/lda.log \
    ivector-compute-lda --total-covariance-factor=0.0 --dim=$lda_dim \
    "ark:ivector-subtract-global-mean scp:$nnet_dir/xvectors_train_combined_200k/xvector.scp ark:- |" \
    ark:data/train_combined_200k/utt2spk $nnet_dir/xvectors_train_combined_200k/transform.mat || exit 1;

  # Train the PLDA model.
  $train_cmd $nnet_dir/xvectors_train_combined_200k/log/plda.log \
    ivector-compute-plda ark:data/train_combined_200k/spk2utt \
    "ark:ivector-subtract-global-mean scp:$nnet_dir/xvectors_train_combined_200k/xvector.scp ark:- | transform-vec $nnet_dir/xvectors_train_combined_200k/transform.mat ark:- ark:- | ivector-normalize-length ark:-  ark:- |" \
    $nnet_dir/xvectors_train_combined_200k/plda || exit 1;
fi

if [ $stage -le 11 ]; then
  # Compute PLDA scores for SITW dev core-core trials
  $train_cmd $nnet_dir/scores/log/sitw_dev_core_scoring.log \
    ivector-plda-scoring --normalize-length=true \
    --num-utts=ark:$nnet_dir/xvectors_sitw_dev_enroll/num_utts.ark \
    "ivector-copy-plda --smoothing=0.0 $nnet_dir/xvectors_train_combined_200k/plda - |" \
    "ark:ivector-mean ark:data/sitw_dev_enroll/spk2utt scp:$nnet_dir/xvectors_sitw_dev_enroll/xvector.scp ark:- | ivector-subtract-global-mean $nnet_dir/xvectors_train_combined_200k/mean.vec ark:- ark:- | transform-vec $nnet_dir/xvectors_train_combined_200k/transform.mat ark:- ark:- | ivector-normalize-length ark:- ark:- |" \
    "ark:ivector-subtract-global-mean $nnet_dir/xvectors_train_combined_200k/mean.vec scp:$nnet_dir/xvectors_sitw_dev_test/xvector.scp ark:- | transform-vec $nnet_dir/xvectors_train_combined_200k/transform.mat ark:- ark:- | ivector-normalize-length ark:- ark:- |" \
    "cat '$sitw_dev_trials_core' | cut -d\  --fields=1,2 |" $nnet_dir/scores/sitw_dev_core_scores || exit 1;

  # SITW Dev Core:
  # EER: 3.003%
  # minDCF(p-target=0.01): 0.3119
  # minDCF(p-target=0.001): 0.4955
  echo "SITW Dev Core:"
  eer=$(paste $sitw_dev_trials_core $nnet_dir/scores/sitw_dev_core_scores | awk '{print $6, $3}' | compute-eer - 2>/dev/null)
  mindcf1=`sid/compute_min_dcf.py --p-target 0.01 $nnet_dir/scores/sitw_dev_core_scores $sitw_dev_trials_core 2> /dev/null`
  mindcf2=`sid/compute_min_dcf.py --p-target 0.001 $nnet_dir/scores/sitw_dev_core_scores $sitw_dev_trials_core 2> /dev/null`
  echo "EER: $eer%"
  echo "minDCF(p-target=0.01): $mindcf1"
  echo "minDCF(p-target=0.001): $mindcf2"
fi

if [ $stage -le 12 ]; then
  # Compute PLDA scores for SITW eval core-core trials
  $train_cmd $nnet_dir/scores/log/sitw_eval_core_scoring.log \
    ivector-plda-scoring --normalize-length=true \
    --num-utts=ark:$nnet_dir/xvectors_sitw_eval_enroll/num_utts.ark \
    "ivector-copy-plda --smoothing=0.0 $nnet_dir/xvectors_train_combined_200k/plda - |" \
    "ark:ivector-mean ark:data/sitw_eval_enroll/spk2utt scp:$nnet_dir/xvectors_sitw_eval_enroll/xvector.scp ark:- | ivector-subtract-global-mean $nnet_dir/xvectors_train_combined_200k/mean.vec ark:- ark:- | transform-vec $nnet_dir/xvectors_train_combined_200k/transform.mat ark:- ark:- | ivector-normalize-length ark:- ark:- |" \
    "ark:ivector-subtract-global-mean $nnet_dir/xvectors_train_combined_200k/mean.vec scp:$nnet_dir/xvectors_sitw_eval_test/xvector.scp ark:- | transform-vec $nnet_dir/xvectors_train_combined_200k/transform.mat ark:- ark:- | ivector-normalize-length ark:- ark:- |" \
    "cat '$sitw_eval_trials_core' | cut -d\  --fields=1,2 |" $nnet_dir/scores/sitw_eval_core_scores || exit 1;

  # SITW Eval Core:
  # EER: 3.499%
  # minDCF(p-target=0.01): 0.3424
  # minDCF(p-target=0.001): 0.5164
  echo -e "\nSITW Eval Core:";
  eer=$(paste $sitw_eval_trials_core $nnet_dir/scores/sitw_eval_core_scores | awk '{print $6, $3}' | compute-eer - 2>/dev/null)
  mindcf1=`sid/compute_min_dcf.py --p-target 0.01 $nnet_dir/scores/sitw_eval_core_scores $sitw_eval_trials_core 2> /dev/null`
  mindcf2=`sid/compute_min_dcf.py --p-target 0.001 $nnet_dir/scores/sitw_eval_core_scores $sitw_eval_trials_core 2> /dev/null`
  echo "EER: $eer%"
  echo "minDCF(p-target=0.01): $mindcf1"
  echo "minDCF(p-target=0.001): $mindcf2"
fi
