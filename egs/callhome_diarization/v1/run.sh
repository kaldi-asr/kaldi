#!/usr/bin/env bash
# Copyright 2017-2018  David Snyder
#           2017-2018  Matthew Maciejewski
# Apache 2.0.
#
# This is still a work in progress, but implements something similar to
# Greg Sell's and Daniel Garcia-Romero's iVector-based diarization system
# in 'Speaker Diarization With PLDA I-Vector Scoring And Unsupervised
# Calibration'.  The main difference is that we haven't implemented the
# VB resegmentation yet.

. ./cmd.sh
. ./path.sh
set -e
mfccdir=`pwd`/mfcc
vaddir=`pwd`/mfcc
data_root=/export/corpora5/LDC
num_components=2048
ivector_dim=128
stage=0

# Prepare datasets
if [ $stage -le 0 ]; then
  # Prepare a collection of NIST SRE data. This will be used to train the UBM,
  # iVector extractor and PLDA model.
  local/make_sre.sh $data_root data

  # Prepare SWB for UBM and iVector extractor training.
  local/make_swbd2_phase2.pl $data_root/LDC99S79 \
                           data/swbd2_phase2_train
  local/make_swbd2_phase3.pl $data_root/LDC2002S06 \
                           data/swbd2_phase3_train
  local/make_swbd_cellular1.pl $data_root/LDC2001S13 \
                             data/swbd_cellular1_train
  local/make_swbd_cellular2.pl $data_root/LDC2004S07 \
                             data/swbd_cellular2_train

  # Prepare the Callhome portion of NIST SRE 2000.
  local/make_callhome.sh /export/corpora/NIST/LDC2001S97/ data/

  utils/combine_data.sh data/train \
    data/swbd_cellular1_train data/swbd_cellular2_train \
    data/swbd2_phase2_train data/swbd2_phase3_train data/sre
fi

# Prepare features
if [ $stage -le 1 ]; then
  # The script local/make_callhome.sh splits callhome into two parts, called
  # callhome1 and callhome2.  Each partition is treated like a held-out
  # dataset, and used to estimate various quantities needed to perform
  # diarization on the other part (and vice versa).
  for name in train callhome1 callhome2; do
    steps/make_mfcc.sh --mfcc-config conf/mfcc.conf --nj 40 \
      --cmd "$train_cmd" --write-utt2num-frames true \
      data/$name exp/make_mfcc $mfccdir
    utils/fix_data_dir.sh data/$name
  done

  for name in train callhome1 callhome2; do
    sid/compute_vad_decision.sh --nj 40 --cmd "$train_cmd" \
      data/$name exp/make_vad $vaddir
    utils/fix_data_dir.sh data/$name
  done

  # The sre dataset is a subset of train
  cp data/train/{feats,vad}.scp data/sre/
  utils/fix_data_dir.sh data/sre

  # Create segments for ivector extraction for PLDA training data.
  echo "0.01" > data/sre/frame_shift
  diarization/vad_to_segments.sh --nj 40 --cmd "$train_cmd" \
    data/sre data/sre_segmented
fi

# Train UBM and i-vector extractor
if [ $stage -le 2 ]; then
  # Reduce the amount of training data for the UBM.
  utils/subset_data_dir.sh data/train 16000 data/train_16k
  utils/subset_data_dir.sh data/train 32000 data/train_32k

  # Train UBM and i-vector extractor.
  sid/train_diag_ubm.sh --cmd "$train_cmd --mem 20G" \
    --nj 40 --num-threads 8 --delta-order 1 --apply-cmn false \
    data/train_16k $num_components exp/diag_ubm_$num_components

  sid/train_full_ubm.sh --nj 40 --remove-low-count-gaussians false \
    --cmd "$train_cmd --mem 25G" --apply-cmn false \
    data/train_32k exp/diag_ubm_$num_components \
    exp/full_ubm_$num_components

  sid/train_ivector_extractor.sh \
    --cmd "$train_cmd --mem 35G" \
    --ivector-dim $ivector_dim --num-iters 5 --apply-cmn false \
    exp/full_ubm_$num_components/final.ubm data/train \
    exp/extractor_c${num_components}_i${ivector_dim}
fi

# Extract i-vectors
if [ $stage -le 3 ]; then
  # Reduce the amount of training data for the PLDA,
  utils/subset_data_dir.sh data/sre_segmented 128000 data/sre_segmented_128k
  # Extract iVectors for the SRE, which is our PLDA training
  # data.  A long period is used here so that we don't compute too
  # many iVectors for each recording.
  diarization/extract_ivectors.sh --cmd "$train_cmd --mem 25G" \
    --nj 40 --window 3.0 --period 10.0 --min-segment 1.5 --apply-cmn false \
    --hard-min true exp/extractor_c${num_components}_i${ivector_dim} \
    data/sre_segmented_128k exp/ivectors_sre_segmented_128k

  # Extract iVectors for the two partitions of callhome.
  diarization/extract_ivectors.sh --cmd "$train_cmd --mem 20G" \
    --nj 40 --window 1.5 --period 0.75 --apply-cmn false \
    --min-segment 0.5 exp/extractor_c${num_components}_i${ivector_dim} \
    data/callhome1 exp/ivectors_callhome1

  diarization/extract_ivectors.sh --cmd "$train_cmd --mem 20G" \
    --nj 40 --window 1.5 --period 0.75 --apply-cmn false \
    --min-segment 0.5 exp/extractor_c${num_components}_i${ivector_dim} \
    data/callhome2 exp/ivectors_callhome2
fi

# Train PLDA models
if [ $stage -le 4 ]; then
  # Train a PLDA model on SRE, using callhome1 to whiten.
  # We will later use this to score iVectors in callhome2.
  "$train_cmd" exp/ivectors_callhome1/log/plda.log \
    ivector-compute-plda ark:exp/ivectors_sre_segmented_128k/spk2utt \
      "ark:ivector-subtract-global-mean \
      scp:exp/ivectors_sre_segmented_128k/ivector.scp ark:- \
      | transform-vec exp/ivectors_callhome1/transform.mat ark:- ark:- \
      | ivector-normalize-length ark:- ark:- |" \
    exp/ivectors_callhome1/plda || exit 1;

  # Train a PLDA model on SRE, using callhome2 to whiten.
  # We will later use this to score iVectors in callhome1.
  "$train_cmd" exp/ivectors_callhome2/log/plda.log \
    ivector-compute-plda ark:exp/ivectors_sre_segmented_128k/spk2utt \
      "ark:ivector-subtract-global-mean \
      scp:exp/ivectors_sre_segmented_128k/ivector.scp ark:- \
      | transform-vec exp/ivectors_callhome2/transform.mat ark:- ark:- \
      | ivector-normalize-length ark:- ark:- |" \
    exp/ivectors_callhome2/plda || exit 1;
fi

# Perform PLDA scoring
if [ $stage -le 5 ]; then
  # Perform PLDA scoring on all pairs of segments for each recording.
  # The first directory contains the PLDA model that used callhome2
  # to perform whitening (recall that we're treating callhome2 as a
  # held-out dataset).  The second directory contains the iVectors
  # for callhome1.
  diarization/score_plda.sh --cmd "$train_cmd --mem 4G" \
    --nj 20 exp/ivectors_callhome2 exp/ivectors_callhome1 \
    exp/ivectors_callhome1/plda_scores

  # Do the same thing for callhome2.
  diarization/score_plda.sh --cmd "$train_cmd --mem 4G" \
    --nj 20 exp/ivectors_callhome1 exp/ivectors_callhome2 \
    exp/ivectors_callhome2/plda_scores
fi

# Cluster the PLDA scores using a stopping threshold.
if [ $stage -le 6 ]; then
  # First, we find the threshold that minimizes the DER on each partition of
  # callhome.
  mkdir -p exp/tuning
  for dataset in callhome1 callhome2; do
    echo "Tuning clustering threshold for $dataset"
    best_der=100
    best_threshold=0
    utils/filter_scp.pl -f 2 data/$dataset/wav.scp \
      data/callhome/fullref.rttm > data/$dataset/ref.rttm

    # The threshold is in terms of the log likelihood ratio provided by the
    # PLDA scores.  In a perfectly calibrated system, the threshold is 0.
    # In the following loop, we evaluate the clustering on a heldout dataset
    # (callhome1 is heldout for callhome2 and vice-versa) using some reasonable
    # thresholds for a well-calibrated system.
    for threshold in -0.3 -0.2 -0.1 -0.05 0 0.05 0.1 0.2 0.3; do
      diarization/cluster.sh --cmd "$train_cmd --mem 4G" --nj 20 \
        --threshold $threshold exp/ivectors_$dataset/plda_scores \
        exp/ivectors_$dataset/plda_scores_t$threshold

      md-eval.pl -1 -c 0.25 -r data/$dataset/ref.rttm \
       -s exp/ivectors_$dataset/plda_scores_t$threshold/rttm \
       2> exp/tuning/${dataset}_t${threshold}.log \
       > exp/tuning/${dataset}_t${threshold}

      der=$(grep -oP 'DIARIZATION\ ERROR\ =\ \K[0-9]+([.][0-9]+)?' \
        exp/tuning/${dataset}_t${threshold})
      if [ $(perl -e "print ($der < $best_der ? 1 : 0);") -eq 1 ]; then
        best_der=$der
        best_threshold=$threshold
      fi
    done
    echo "$best_threshold" > exp/tuning/${dataset}_best
  done

  # Cluster callhome1 using the best threshold found for callhome2.  This way,
  # callhome2 is treated as a held-out dataset to discover a reasonable
  # stopping threshold for callhome1.
  diarization/cluster.sh --cmd "$train_cmd --mem 4G" --nj 20 \
    --threshold $(cat exp/tuning/callhome2_best) \
    exp/ivectors_callhome1/plda_scores exp/ivectors_callhome1/plda_scores

  # Do the same thing for callhome2, treating callhome1 as a held-out dataset
  # to discover a stopping threshold.
  diarization/cluster.sh --cmd "$train_cmd --mem 4G" --nj 20 \
    --threshold $(cat exp/tuning/callhome1_best) \
    exp/ivectors_callhome2/plda_scores exp/ivectors_callhome2/plda_scores

  mkdir -p exp/results
  # Now combine the results for callhome1 and callhome2 and evaluate it
  # together.
  cat exp/ivectors_callhome1/plda_scores/rttm \
    exp/ivectors_callhome2/plda_scores/rttm | md-eval.pl -1 -c 0.25 -r \
    data/callhome/fullref.rttm -s - 2> exp/results/threshold.log \
    > exp/results/DER_threshold.txt
  der=$(grep -oP 'DIARIZATION\ ERROR\ =\ \K[0-9]+([.][0-9]+)?' \
    exp/results/DER_threshold.txt)
  # Using supervised calibration, DER: 10.36%
  echo "Using supervised calibration, DER: $der%"
fi

# Cluster the PLDA scores using the oracle number of speakers
if [ $stage -le 7 ]; then
  # In this section, we show how to do the clustering if the number of speakers
  # (and therefore, the number of clusters) per recording is known in advance.
  diarization/cluster.sh --cmd "$train_cmd --mem 4G" \
    --reco2num-spk data/callhome1/reco2num_spk \
    exp/ivectors_callhome1/plda_scores exp/ivectors_callhome1/plda_scores_num_spk

  diarization/cluster.sh --cmd "$train_cmd --mem 4G" \
    --reco2num-spk data/callhome2/reco2num_spk \
    exp/ivectors_callhome2/plda_scores exp/ivectors_callhome2/plda_scores_num_spk

  mkdir -p exp/results
  # Now combine the results for callhome1 and callhome2 and evaluate it together.
  cat exp/ivectors_callhome1/plda_scores_num_spk/rttm \
  exp/ivectors_callhome2/plda_scores_num_spk/rttm \
    | md-eval.pl -1 -c 0.25 -r data/callhome/fullref.rttm -s - 2> exp/results/num_spk.log \
    > exp/results/DER_num_spk.txt
  der=$(grep -oP 'DIARIZATION\ ERROR\ =\ \K[0-9]+([.][0-9]+)?' \
    exp/results/DER_num_spk.txt)
  # Using the oracle number of speakers, DER: 8.69%
  echo "Using the oracle number of speakers, DER: $der%"
fi
