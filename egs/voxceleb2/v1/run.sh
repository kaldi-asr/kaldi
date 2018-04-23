#!/bin/bash
# Copyright      2017   David Snyder
#                2017   Johns Hopkins University (Author: Daniel Garcia-Romero)
#                2017   Johns Hopkins University (Author: Daniel Povey)
#                2018   Ewald Enzinger
# Apache 2.0.
#
# Adapted from SRE16 v1 recipe (commit 3ea534070fd2cccd2e4ee21772132230033022ce)
#
# See ../README.txt for more info on data required.
# Results (mostly EERs) are inline in comments below.

. ./cmd.sh
. ./path.sh
set -e
mfccdir=`pwd`/mfcc
vaddir=`pwd`/mfcc

voxceleb2_trials=data/voxceleb2_test/trials
voxceleb2_root=/path/to/voxceleb2

stage=0

. utils/parse_options.sh

if [ $stage -le 0 ]; then
  local/make_voxceleb2.pl $voxceleb2_root dev data/voxceleb2_train
  local/make_voxceleb2.pl $voxceleb2_root test data/voxceleb2_test
fi

if [ $stage -le 1 ]; then
  # Make MFCCs and compute the energy-based VAD for each dataset
  for name in voxceleb2_train voxceleb2_test; do
    steps/make_mfcc.sh --mfcc-config conf/mfcc.conf --nj 40 --cmd "$train_cmd" \
      data/${name} exp/make_mfcc $mfccdir
    utils/fix_data_dir.sh data/${name}
    sid/compute_vad_decision.sh --nj 40 --cmd "$train_cmd" \
      data/${name} exp/make_vad $vaddir
    utils/fix_data_dir.sh data/${name}
  done
fi

if [ $stage -le 2 ]; then
  # Train the UBM.
  sid/train_diag_ubm.sh --cmd "$train_cmd --mem 4G" \
    --nj 40 --num-threads 8 --subsample 1 \
    data/voxceleb2_train 2048 \
    exp/diag_ubm

  sid/train_full_ubm.sh --cmd "$train_cmd --mem 25G" \
    --nj 40 --remove-low-count-gaussians false --subsample 1 \
    data/voxceleb2_train \
    exp/diag_ubm exp/full_ubm
fi

if [ $stage -le 3 ]; then
  # Train the i-vector extractor.
  sid/train_ivector_extractor.sh --cmd "$train_cmd --mem 20G" \
    --ivector-dim 400 --num-iters 5 --stage 1 \
    exp/full_ubm/final.ubm data/voxceleb2_train \
    exp/extractor
fi

if [ $stage -le 4 ]; then
  sid/extract_ivectors.sh --cmd "$train_cmd --mem 4G" --nj 40 \
    exp/extractor data/voxceleb2_train \
    exp/ivectors_voxceleb2_train

  sid/extract_ivectors.sh --cmd "$train_cmd --mem 4G" --nj 40 \
    exp/extractor data/voxceleb2_test \
    exp/ivectors_voxceleb2_test
fi

if [ $stage -le 5 ]; then
  local/produce_trials.py data/voxceleb2_test/utt2spk $voxceleb2_trials
fi

if [ $stage -le 6 ]; then
  # Compute the mean vector for centering the evaluation i-vectors.
  $train_cmd exp/ivectors_voxceleb2_train/log/compute_mean.log \
    ivector-mean scp:exp/ivectors_voxceleb2_train/ivector.scp \
    exp/ivectors_voxceleb2_train/mean.vec || exit 1;

  # This script uses LDA to decrease the dimensionality prior to PLDA.
  lda_dim=200
  $train_cmd exp/ivectors_voxceleb2_train/log/lda.log \
    ivector-compute-lda --total-covariance-factor=0.0 --dim=$lda_dim \
    "ark:ivector-subtract-global-mean scp:exp/ivectors_voxceleb2_train/ivector.scp ark:- |" \
    ark:data/voxceleb2_train/utt2spk exp/ivectors_voxceleb2_train/transform.mat || exit 1;

  #  Train the PLDA model.
  $train_cmd exp/ivectors_voxceleb2_train/log/plda.log \
    ivector-compute-plda ark:data/voxceleb2_train/spk2utt \
    "ark:ivector-subtract-global-mean scp:exp/ivectors_voxceleb2_train/ivector.scp ark:- | transform-vec exp/ivectors_voxceleb2_train/transform.mat ark:- ark:- | ivector-normalize-length ark:-  ark:- |" \
    exp/ivectors_voxceleb2_train/plda || exit 1;
fi

if [ $stage -le 7 ]; then
  # Treat each test utterance independently from the other test utterances of the same speaker
  awk '{print $1, 1;}' <data/voxceleb2_test/utt2spk >exp/ivectors_voxceleb2_test/num_utts.ark

  $train_cmd exp/scores/log/voxceleb2_test_scoring.log \
    ivector-plda-scoring --normalize-length=true \
    --num-utts=ark:exp/ivectors_voxceleb2_test/num_utts.ark \
    "ivector-copy-plda --smoothing=0.0 exp/ivectors_voxceleb2_train/plda - |" \
    "ark:ivector-subtract-global-mean exp/ivectors_voxceleb2_train/mean.vec scp:exp/ivectors_voxceleb2_test/ivector.scp ark:- | transform-vec exp/ivectors_voxceleb2_train/transform.mat ark:- ark:- | ivector-normalize-length ark:- ark:- |" \
    "ark:ivector-subtract-global-mean exp/ivectors_voxceleb2_train/mean.vec scp:exp/ivectors_voxceleb2_test/ivector.scp ark:- | transform-vec exp/ivectors_voxceleb2_train/transform.mat ark:- ark:- | ivector-normalize-length ark:- ark:- |" \
    "cat '$voxceleb2_trials' | cut -d\  --fields=1,2 |" exp/scores_voxceleb2_test || exit 1;
fi

if [ $stage -le 8 ]; then
  eer=`compute-eer <(awk '{if(substr($1,1,7) == substr($2,1,7)) {print $3, "target";} else {print $3, "nontarget";}}' exp/scores_voxceleb2_test) 2> /dev/null`
  echo "EER: ${eer}%"
  # EER: 7.204%
fi
