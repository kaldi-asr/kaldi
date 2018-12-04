#!/bin/bash
# Copyright   2017   Johns Hopkins University (Author: Daniel Garcia-Romero)
#             2017   Johns Hopkins University (Author: Daniel Povey)
#        2017-2018   David Snyder
#             2018   Ewald Enzinger
#             2018   Zili Huang
# Apache 2.0.
#
# See ../README.txt for more info on data required.
# Results (diarization error rate) are inline in comments below.

. ./cmd.sh
. ./path.sh
set -e
mfccdir=`pwd`/mfcc
vaddir=`pwd`/mfcc

voxceleb1_root=/export/corpora/VoxCeleb1
voxceleb2_root=/export/corpora/VoxCeleb2
nnet_dir=exp/xvector_nnet_1a
musan_root=/export/corpora/JHU/musan
dihard_2018_dev=/export/corpora/LDC/LDC2018E31
dihard_2018_eval=/export/corpora/LDC/LDC2018E32v1.1

stage=0

if [ $stage -le 0 ]; then
  local/make_voxceleb2.pl $voxceleb2_root dev data/voxceleb2_train
  local/make_voxceleb2.pl $voxceleb2_root test data/voxceleb2_test
  # This script creates data/voxceleb1_test and data/voxceleb1_train.
  # Our evaluation set is the test portion of VoxCeleb1.
  local/make_voxceleb1.pl $voxceleb1_root data
  # We'll train on all of VoxCeleb2, plus the training portion of VoxCeleb1.
  # This should give 7,351 speakers and 1,277,503 utterances.
  utils/combine_data.sh data/train data/voxceleb2_train data/voxceleb2_test data/voxceleb1_train

  # Prepare the development and evaluation set for DIHARD 2018.
  local/make_dihard_2018_dev.sh $dihard_2018_dev data/dihard_2018_dev
  local/make_dihard_2018_eval.sh $dihard_2018_eval data/dihard_2018_eval
fi

if [ $stage -le 1 ]; then
  # Make MFCCs for each dataset.
  for name in train dihard_2018_dev dihard_2018_eval; do
    steps/make_mfcc.sh --write-utt2num-frames true --mfcc-config conf/mfcc.conf --nj 40 --cmd "$train_cmd --max-jobs-run 20" \
      data/${name} exp/make_mfcc $mfccdir
    utils/fix_data_dir.sh data/${name}
  done

  # Compute the energy-based VAD for training set.
  sid/compute_vad_decision.sh --nj 40 --cmd "$train_cmd" \
      data/train exp/make_vad $vaddir
  utils/fix_data_dir.sh data/train

  # This writes features to disk after applying the sliding window CMN.
  # Although this is somewhat wasteful in terms of disk space, for diarization
  # it ends up being preferable to performing the CMN in memory.  If the CMN
  # were performed in memory (e.g., we used --apply-cmn true in
  # diarization/nnet3/xvector/extract_xvectors.sh) it would need to be
  # performed after the subsegmentation, which leads to poorer results.
  for name in train dihard_2018_dev dihard_2018_eval; do
    local/nnet3/xvector/prepare_feats.sh --nj 40 --cmd "$train_cmd" \
      data/$name data/${name}_cmn exp/${name}_cmn
    if [ -f data/$name/vad.scp ]; then
      cp data/$name/vad.scp data/${name}_cmn/
    fi
    if [ -f data/$name/segments ]; then
      cp data/$name/segments data/${name}_cmn/
    fi
    utils/fix_data_dir.sh data/${name}_cmn
  done

  echo "0.01" > data/train_cmn/frame_shift
  # Create segments to extract x-vectors from for PLDA training data.
  # The segments are created using an energy-based speech activity
  # detection (SAD) system, but this is not necessary.  You can replace
  # this with segments computed from your favorite SAD.
  diarization/vad_to_segments.sh --nj 40 --cmd "$train_cmd" \
      data/train_cmn data/train_cmn_segmented
fi

# In this section, we augment the training data with reverberation,
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

  # Make a reverberated version of the training data.  Note that we don't add any
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
  local/make_musan.sh $musan_root data

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
  steps/make_mfcc.sh --mfcc-config conf/mfcc.conf --nj 40 --cmd "$train_cmd --max-jobs-run 20" \
    data/train_aug_1m exp/make_mfcc $mfccdir

  # Combine the clean and augmented training data.  This is now roughly
  # double the size of the original clean list.
  utils/combine_data.sh data/train_combined data/train_aug_1m data/train
fi

# Now we prepare the features to generate examples for xvector training.
if [ $stage -le 4 ]; then
  # This script applies CMVN and removes nonspeech frames.  Note that this is somewhat
  # wasteful, as it roughly doubles the amount of training data on disk.  After
  # creating training examples, this can be removed.
  local/nnet3/xvector/prepare_feats_for_egs.sh --nj 40 --cmd "$train_cmd" \
    data/train_combined data/train_combined_no_sil exp/train_combined_no_sil
  utils/fix_data_dir.sh data/train_combined_no_sil
fi

if [ $stage -le 5 ]; then
  # Now, we need to remove features that are too short after removing silence
  # frames.  We want at least 4s (400 frames) per utterance.
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

# Stages 6 through 8 are handled in run_xvector.sh, a TDNN embedding extractor is trained.
local/nnet3/xvector/run_xvector.sh --stage $stage --train-stage -1 \
  --data data/train_combined_no_sil --nnet-dir $nnet_dir \
  --egs-dir $nnet_dir/egs

if [ $stage -le 9 ]; then
  # Extract x-vectors for DIHARD 2018 development and evaluation set.
  diarization/nnet3/xvector/extract_xvectors.sh --cmd "$train_cmd --mem 5G" \
    --nj 40 --window 1.5 --period 0.75 --apply-cmn false \
    --min-segment 0.5 $nnet_dir \
    data/dihard_2018_dev_cmn $nnet_dir/xvectors_dihard_2018_dev

  diarization/nnet3/xvector/extract_xvectors.sh --cmd "$train_cmd --mem 5G" \
    --nj 40 --window 1.5 --period 0.75 --apply-cmn false \
    --min-segment 0.5 $nnet_dir \
    data/dihard_2018_eval_cmn $nnet_dir/xvectors_dihard_2018_eval

  # Reduce the amount of training data for the PLDA training.
  utils/subset_data_dir.sh data/train_cmn_segmented 128000 data/train_cmn_segmented_128k
  # Extract x-vectors for the VoxCeleb, which is our PLDA training
  # data.  A long period is used here so that we don't compute too
  # many x-vectors for each recording.
  diarization/nnet3/xvector/extract_xvectors.sh --cmd "$train_cmd --mem 10G" \
    --nj 40 --window 3.0 --period 10.0 --min-segment 1.5 --apply-cmn false \
    --hard-min true $nnet_dir \
    data/train_cmn_segmented_128k $nnet_dir/xvectors_train_segmented_128k
fi

# Train PLDA models
if [ $stage -le 10 ]; then
  # Train a PLDA model on VoxCeleb, using DIHARD 2018 development set to whiten.
  "$train_cmd" $nnet_dir/xvectors_dihard_2018_dev/log/plda.log \
    ivector-compute-plda ark:$nnet_dir/xvectors_train_segmented_128k/spk2utt \
      "ark:ivector-subtract-global-mean \
      scp:$nnet_dir/xvectors_train_segmented_128k/xvector.scp ark:- \
      | transform-vec $nnet_dir/xvectors_dihard_2018_dev/transform.mat ark:- ark:- \
      | ivector-normalize-length ark:- ark:- |" \
    $nnet_dir/xvectors_dihard_2018_dev/plda || exit 1;
fi

# Perform PLDA scoring
if [ $stage -le 11 ]; then
  # Perform PLDA scoring on all pairs of segments for each recording.
  diarization/nnet3/xvector/score_plda.sh --cmd "$train_cmd --mem 4G" \
    --nj 20 $nnet_dir/xvectors_dihard_2018_dev $nnet_dir/xvectors_dihard_2018_dev \
    $nnet_dir/xvectors_dihard_2018_dev/plda_scores

  diarization/nnet3/xvector/score_plda.sh --cmd "$train_cmd --mem 4G" \
    --nj 20 $nnet_dir/xvectors_dihard_2018_dev $nnet_dir/xvectors_dihard_2018_eval \
    $nnet_dir/xvectors_dihard_2018_eval/plda_scores
fi

# Cluster the PLDA scores using a stopping threshold.
if [ $stage -le 12 ]; then
  # First, we find the threshold that minimizes the DER on DIHARD 2018 development set.
  mkdir -p $nnet_dir/tuning
  echo "Tuning clustering threshold for DIHARD 2018 development set"
  best_der=100
  best_threshold=0

  # The threshold is in terms of the log likelihood ratio provided by the
  # PLDA scores.  In a perfectly calibrated system, the threshold is 0.
  # In the following loop, we evaluate DER performance on DIHARD 2018 development 
  # set using some reasonable thresholds for a well-calibrated system.
  for threshold in -0.5 -0.4 -0.3 -0.2 -0.1 -0.05 0 0.05 0.1 0.2 0.3 0.4 0.5; do
    diarization/cluster.sh --cmd "$train_cmd --mem 4G" --nj 20 \
      --threshold $threshold --rttm-channel 1 $nnet_dir/xvectors_dihard_2018_dev/plda_scores \
      $nnet_dir/xvectors_dihard_2018_dev/plda_scores_t$threshold

    md-eval.pl -r data/dihard_2018_dev/rttm \
     -s $nnet_dir/xvectors_dihard_2018_dev/plda_scores_t$threshold/rttm \
     2> $nnet_dir/tuning/dihard_2018_dev_t${threshold}.log \
     > $nnet_dir/tuning/dihard_2018_dev_t${threshold}

    der=$(grep -oP 'DIARIZATION\ ERROR\ =\ \K[0-9]+([.][0-9]+)?' \
      $nnet_dir/tuning/dihard_2018_dev_t${threshold})
    if [ $(echo $der'<'$best_der | bc -l) -eq 1 ]; then
      best_der=$der
      best_threshold=$threshold
    fi
  done
  echo "$best_threshold" > $nnet_dir/tuning/dihard_2018_dev_best

  diarization/cluster.sh --cmd "$train_cmd --mem 4G" --nj 20 \
    --threshold $(cat $nnet_dir/tuning/dihard_2018_dev_best) --rttm-channel 1 \
    $nnet_dir/xvectors_dihard_2018_dev/plda_scores $nnet_dir/xvectors_dihard_2018_dev/plda_scores

  # Cluster DIHARD 2018 evaluation set using the best threshold found for the DIHARD 
  # 2018 development set. The DIHARD 2018 development set is used as the validation 
  # set to tune the parameters. 
  diarization/cluster.sh --cmd "$train_cmd --mem 4G" --nj 20 \
    --threshold $(cat $nnet_dir/tuning/dihard_2018_dev_best) --rttm-channel 1 \
    $nnet_dir/xvectors_dihard_2018_eval/plda_scores $nnet_dir/xvectors_dihard_2018_eval/plda_scores

  mkdir -p $nnet_dir/results
  # Compute the DER on the DIHARD 2018 evaluation set. We use the official metrics of   
  # the DIHARD challenge. The DER is calculated with no unscored collars and including  
  # overlapping speech.
  md-eval.pl -r data/dihard_2018_eval/rttm \
    -s $nnet_dir/xvectors_dihard_2018_eval/plda_scores/rttm 2> $nnet_dir/results/threshold.log \
    > $nnet_dir/results/DER_threshold.txt
  der=$(grep -oP 'DIARIZATION\ ERROR\ =\ \K[0-9]+([.][0-9]+)?' \
    $nnet_dir/results/DER_threshold.txt)
  # Using supervised calibration, DER: 26.47%
  echo "Using supervised calibration, DER: $der%"
fi

# Cluster the PLDA scores using the oracle number of speakers
if [ $stage -le 13 ]; then
  # In this section, we show how to do the clustering if the number of speakers
  # (and therefore, the number of clusters) per recording is known in advance.
  diarization/cluster.sh --cmd "$train_cmd --mem 4G" --nj 20 \
    --reco2num-spk data/dihard_2018_eval/reco2num_spk --rttm-channel 1 \
    $nnet_dir/xvectors_dihard_2018_eval/plda_scores $nnet_dir/xvectors_dihard_2018_eval/plda_scores_num_spk

  md-eval.pl -r data/dihard_2018_eval/rttm \
    -s $nnet_dir/xvectors_dihard_2018_eval/plda_scores_num_spk/rttm 2> $nnet_dir/results/num_spk.log \
    > $nnet_dir/results/DER_num_spk.txt
  der=$(grep -oP 'DIARIZATION\ ERROR\ =\ \K[0-9]+([.][0-9]+)?' \
    $nnet_dir/results/DER_num_spk.txt)
  # Using the oracle number of speakers, DER: 23.90%
  echo "Using the oracle number of speakers, DER: $der%"
fi
