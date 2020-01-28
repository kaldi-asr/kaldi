#!/usr/bin/env bash
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
dihard_2018_dev=/export/corpora/LDC/LDC2018E31
dihard_2018_eval=/export/corpora/LDC/LDC2018E32v1.1
num_components=2048
ivector_dim=400
ivec_dir=exp/extractor_c${num_components}_i${ivector_dim}

stage=0

if [ $stage -le 0 ]; then
  local/make_voxceleb2.pl $voxceleb2_root dev data/voxceleb2_train
  local/make_voxceleb2.pl $voxceleb2_root test data/voxceleb2_test

  # Now prepare the VoxCeleb1 train and test data.  If you downloaded the corpus soon
  # after it was first released, you may need to use an older version of the script, which
  # can be invoked as follows:
  # local/make_voxceleb1.pl $voxceleb1_root data
  local/make_voxceleb1_v2.pl $voxceleb1_root dev data/voxceleb1_train
  local/make_voxceleb1_v2.pl $voxceleb1_root test data/voxceleb1_test

  # We'll train on all of VoxCeleb2, plus the training portion of VoxCeleb1.
  # This should give 7,351 speakers and 1,277,503 utterances.
  utils/combine_data.sh data/train data/voxceleb2_train data/voxceleb2_test data/voxceleb1_train

  # Prepare the development and evaluation set for DIHARD 2018.
  local/make_dihard_2018_dev.sh $dihard_2018_dev data/dihard_2018_dev
  local/make_dihard_2018_eval.sh $dihard_2018_eval data/dihard_2018_eval
fi

if [ $stage -le 1 ]; then
  # Make MFCCs for each dataset
  for name in train dihard_2018_dev dihard_2018_eval; do
    steps/make_mfcc.sh --write-utt2num-frames true \
      --mfcc-config conf/mfcc.conf --nj 40 --cmd "$train_cmd --max-jobs-run 20" \
      data/${name} exp/make_mfcc $mfccdir
    utils/fix_data_dir.sh data/${name}
  done

  # Compute the energy-based VAD for train
  sid/compute_vad_decision.sh --nj 40 --cmd "$train_cmd" \
    data/train exp/make_vad $vaddir
  utils/fix_data_dir.sh data/train

  # This writes features to disk after adding deltas and applying the sliding window CMN.
  # Although this is somewhat wasteful in terms of disk space, for diarization
  # it ends up being preferable to performing the CMN in memory.  If the CMN
  # were performed in memory it would need to be performed after the subsegmentation,
  # which leads to poorer results.
  for name in train dihard_2018_dev dihard_2018_eval; do
    local/prepare_feats.sh --nj 40 --cmd "$train_cmd" \
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
  # Create segments to extract i-vectors from for PLDA training data.
  # The segments are created using an energy-based speech activity
  # detection (SAD) system, but this is not necessary.  You can replace
  # this with segments computed from your favorite SAD.
  diarization/vad_to_segments.sh --nj 40 --cmd "$train_cmd" \
      data/train_cmn data/train_cmn_segmented
fi

if [ $stage -le 2 ]; then
  # Train the UBM on VoxCeleb 1 and 2.
  sid/train_diag_ubm.sh --cmd "$train_cmd --mem 4G" \
    --nj 40 --num-threads 8 \
    data/train $num_components \
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
    --ivector-dim $ivector_dim --num-iters 5 \
    exp/full_ubm/final.ubm data/train_100k \
    $ivec_dir
fi

if [ $stage -le 4 ]; then
  # Extract i-vectors for DIHARD 2018 development and evaluation set. 
  # We set apply-cmn false and apply-deltas false because we already add
  # deltas and apply cmn in stage 1.
  diarization/extract_ivectors.sh --cmd "$train_cmd --mem 20G" \
    --nj 40 --window 1.5 --period 0.75 --apply-cmn false --apply-deltas false \
    --min-segment 0.5 $ivec_dir \
    data/dihard_2018_dev_cmn $ivec_dir/ivectors_dihard_2018_dev

  diarization/extract_ivectors.sh --cmd "$train_cmd --mem 20G" \
    --nj 40 --window 1.5 --period 0.75 --apply-cmn false --apply-deltas false \
    --min-segment 0.5 $ivec_dir \
    data/dihard_2018_eval_cmn $ivec_dir/ivectors_dihard_2018_eval

  # Reduce the amount of training data for the PLDA training.
  utils/subset_data_dir.sh data/train_cmn_segmented 128000 data/train_cmn_segmented_128k
  # Extract i-vectors for the VoxCeleb, which is our PLDA training
  # data.  A long period is used here so that we don't compute too
  # many i-vectors for each recording.
  diarization/extract_ivectors.sh --cmd "$train_cmd --mem 25G" \
    --nj 40 --window 3.0 --period 10.0 --min-segment 1.5 --apply-cmn false --apply-deltas false \
    --hard-min true $ivec_dir \
    data/train_cmn_segmented_128k $ivec_dir/ivectors_train_segmented_128k
fi

if [ $stage -le 5 ]; then
  # Train a PLDA model on VoxCeleb, using DIHARD 2018 development set to whiten.
  "$train_cmd" $ivec_dir/ivectors_dihard_2018_dev/log/plda.log \
    ivector-compute-plda ark:$ivec_dir/ivectors_train_segmented_128k/spk2utt \
      "ark:ivector-subtract-global-mean \
      scp:$ivec_dir/ivectors_train_segmented_128k/ivector.scp ark:- \
      | transform-vec $ivec_dir/ivectors_dihard_2018_dev/transform.mat ark:- ark:- \
      | ivector-normalize-length ark:- ark:- |" \
    $ivec_dir/ivectors_dihard_2018_dev/plda || exit 1;
fi

# Perform PLDA scoring
if [ $stage -le 6 ]; then
  # Perform PLDA scoring on all pairs of segments for each recording.
  diarization/score_plda.sh --cmd "$train_cmd --mem 4G" \
    --nj 20 $ivec_dir/ivectors_dihard_2018_dev $ivec_dir/ivectors_dihard_2018_dev \
    $ivec_dir/ivectors_dihard_2018_dev/plda_scores

  diarization/score_plda.sh --cmd "$train_cmd --mem 4G" \
    --nj 20 $ivec_dir/ivectors_dihard_2018_dev $ivec_dir/ivectors_dihard_2018_eval \
    $ivec_dir/ivectors_dihard_2018_eval/plda_scores
fi

# Cluster the PLDA scores using a stopping threshold.
if [ $stage -le 7 ]; then
  # First, we find the threshold that minimizes the DER on DIHARD 2018 development set.
  mkdir -p $ivec_dir/tuning
  echo "Tuning clustering threshold for DIHARD 2018 development set"
  best_der=100
  best_threshold=0

  # The threshold is in terms of the log likelihood ratio provided by the
  # PLDA scores.  In a perfectly calibrated system, the threshold is 0.
  # In the following loop, we evaluate DER performance on DIHARD 2018 development 
  # set using some reasonable thresholds for a well-calibrated system.
  for threshold in -0.5 -0.4 -0.3 -0.2 -0.1 -0.05 0 0.05 0.1 0.2 0.3 0.4 0.5; do
    diarization/cluster.sh --cmd "$train_cmd --mem 4G" --nj 20 \
      --threshold $threshold --rttm-channel 1 $ivec_dir/ivectors_dihard_2018_dev/plda_scores \
      $ivec_dir/ivectors_dihard_2018_dev/plda_scores_t$threshold

    md-eval.pl -r data/dihard_2018_dev/rttm \
     -s $ivec_dir/ivectors_dihard_2018_dev/plda_scores_t$threshold/rttm \
     2> $ivec_dir/tuning/dihard_2018_dev_t${threshold}.log \
     > $ivec_dir/tuning/dihard_2018_dev_t${threshold}

    der=$(grep -oP 'DIARIZATION\ ERROR\ =\ \K[0-9]+([.][0-9]+)?' \
      $ivec_dir/tuning/dihard_2018_dev_t${threshold})
    if [ $(perl -e "print ($der < $best_der ? 1 : 0);") -eq 1 ]; then
      best_der=$der
      best_threshold=$threshold
    fi
  done
  echo "$best_threshold" > $ivec_dir/tuning/dihard_2018_dev_best

  diarization/cluster.sh --cmd "$train_cmd --mem 4G" --nj 20 \
    --threshold $(cat $ivec_dir/tuning/dihard_2018_dev_best) --rttm-channel 1 \
    $ivec_dir/ivectors_dihard_2018_dev/plda_scores $ivec_dir/ivectors_dihard_2018_dev/plda_scores

  # Cluster DIHARD 2018 evaluation set using the best threshold found for the DIHARD 
  # 2018 development set. The DIHARD 2018 development set is used as the validation 
  # set to tune the parameters. 
  diarization/cluster.sh --cmd "$train_cmd --mem 4G" --nj 20 \
    --threshold $(cat $ivec_dir/tuning/dihard_2018_dev_best) --rttm-channel 1 \
    $ivec_dir/ivectors_dihard_2018_eval/plda_scores $ivec_dir/ivectors_dihard_2018_eval/plda_scores

  mkdir -p $ivec_dir/results
  # Compute the DER on the DIHARD 2018 evaluation set. We use the official metrics of   
  # the DIHARD challenge. The DER is calculated with no unscored collars and including  
  # overlapping speech.
  md-eval.pl -r data/dihard_2018_eval/rttm \
    -s $ivec_dir/ivectors_dihard_2018_eval/plda_scores/rttm 2> $ivec_dir/results/threshold.log \
    > $ivec_dir/results/DER_threshold.txt
  der=$(grep -oP 'DIARIZATION\ ERROR\ =\ \K[0-9]+([.][0-9]+)?' \
    $ivec_dir/results/DER_threshold.txt)
  # Using supervised calibration, DER: 28.51%
  echo "Using supervised calibration, DER: $der%"
fi

# Cluster the PLDA scores using the oracle number of speakers
if [ $stage -le 8 ]; then
  # In this section, we show how to do the clustering if the number of speakers
  # (and therefore, the number of clusters) per recording is known in advance.
  diarization/cluster.sh --cmd "$train_cmd --mem 4G" --nj 20 \
    --reco2num-spk data/dihard_2018_eval/reco2num_spk --rttm-channel 1 \
    $ivec_dir/ivectors_dihard_2018_eval/plda_scores $ivec_dir/ivectors_dihard_2018_eval/plda_scores_num_spk

  md-eval.pl -r data/dihard_2018_eval/rttm \
    -s $ivec_dir/ivectors_dihard_2018_eval/plda_scores_num_spk/rttm 2> $ivec_dir/results/num_spk.log \
    > $ivec_dir/results/DER_num_spk.txt
  der=$(grep -oP 'DIARIZATION\ ERROR\ =\ \K[0-9]+([.][0-9]+)?' \
    $ivec_dir/results/DER_num_spk.txt)
  # Using the oracle number of speakers, DER: 24.42%
  echo "Using the oracle number of speakers, DER: $der%"
fi
