#!/bin/bash
# Copyright   2017   Johns Hopkins University (Author: Daniel Garcia-Romero)
#             2017   Johns Hopkins University (Author: Daniel Povey)
#        2017-2018   David Snyder
#             2018   Ewald Enzinger
#
# Apache 2.0.
#
# See ../README.txt for more info on data required.
# Results (mostly equal error rates) are inline in comments below.

. ./cmd.sh
. ./path.sh
set -e
mfccdir=`pwd`/mfcc
mfcchiresdir=`pwd`/mfcc
bnfdir=`pwd`/bnf
vaddir=`pwd`/mfcc

# The trials file is downloaded by local/make_voxceleb1.pl.
voxceleb1_trials=data/voxceleb1_test/trials
voxceleb1_root=/export/corpora/VoxCeleb1
voxceleb2_root=/export/corpora/VoxCeleb2

stage=0

. utils/parse_options.sh

if [ $stage -le 0 ]; then
  # Train BNF nnet3 model using TEDLIUM data
  local/nnet3/bnf/tedlium_train_bnf.sh
fi

if [ $stage -le 1 ]; then
  local/make_voxceleb2.pl $voxceleb2_root dev data/voxceleb2_train
  local/make_voxceleb2.pl $voxceleb2_root test data/voxceleb2_test
  # This script reates data/voxceleb1_test and data/voxceleb1_train.
  # Our evaluation set is the test portion of VoxCeleb1.
  local/make_voxceleb1.pl $voxceleb1_root data
  # We'll train on all of VoxCeleb2, plus the training portion of VoxCeleb1.
  # This should give 7,351 speakers and 1,277,503 utterances.
  utils/combine_data.sh data/train data/voxceleb2_train data/voxceleb2_test data/voxceleb1_train
fi

if [ $stage -le 2 ]; then
  # Make MFCCs and compute the energy-based VAD for each dataset
  for name in train voxceleb1_test; do
    cp -r data/${name} data/${name}_hires
    steps/make_mfcc.sh --mfcc-config conf/mfcc.conf --nj 40 --cmd "$train_cmd" \
      data/${name} exp/make_mfcc $mfccdir
    utils/fix_data_dir.sh data/${name}
    sid/compute_vad_decision.sh --nj 40 --cmd "$train_cmd" \
      data/${name} exp/make_vad $vaddir
    utils/fix_data_dir.sh data/${name}
    steps/make_mfcc.sh --mfcc-config conf/mfcc_hires.conf --nj 40 --cmd "$train_cmd" \
      data/${name}_hires exp/make_mfcc $mfcchiresdir
    steps/compute_cmvn_stats.sh data/${name}_hires exp/cmvn $mfcchiresdir
    utils/fix_data_dir.sh data/${name}_hires
  done
fi

if [ $stage -le 3 ]; then
  # Extract bottleneck features for each dataset
  for name in train voxceleb1_test; do
    steps/nnet3/make_bottleneck_features.sh --nj 40 --cmd "$train_cmd" --use-gpu true \
      tdnn_bn.batchnorm data/${name}_hires data/${name}_bnf \
      exp/nnet3 exp/make_bnf $bnfdir || exit 1;
    cp data/${name}/vad.scp data/${name}_bnf/
    utils/fix_data_dir.sh data/${name}_bnf
  done
fi

if [ $stage -le 4 ]; then
  # Train the UBM.
  sid/train_diag_ubm.sh --cmd "$train_cmd --mem 4G" \
    --nj 40 --num-threads 8 --delta-order 0 --apply-cmn false \
    data/train_bnf 2048 \
    exp/diag_ubm

  sid/train_full_ubm.sh --cmd "$train_cmd --mem 10G" \
    --nj 40 --remove-low-count-gaussians false --apply-cmn false \
    data/train_bnf \
    exp/diag_ubm exp/full_ubm
fi

if [ $stage -le 5 ]; then
  # Train the i-vector extractor.
  local/nnet3/bnf/train_ivector_extractor.sh --cmd "$train_cmd --mem 25G" \
    --ivector-dim 600 --delta-window 3 --delta-order 2 --add-bnf true \
    --num-iters 5 exp/full_ubm/final.ubm data/train data/train_bnf \
    exp/extractor_bnf
fi

if [ $stage -le 6 ]; then
  local/nnet3/bnf/extract_ivectors.sh --cmd "$train_cmd --mem 4G" --nj 40 --add-bnf true \
    exp/extractor_bnf data/train data/train_bnf \
    exp/ivectors_train

  local/nnet3/bnf/extract_ivectors.sh --cmd "$train_cmd --mem 4G" --nj 40 --add-bnf true \
    exp/extractor data/voxceleb1_test data/voxceleb1_test_bnf \
    exp/ivectors_voxceleb1_test
fi

if [ $stage -le 7 ]; then
  # Compute the mean vector for centering the evaluation i-vectors.
  $train_cmd exp/ivectors_train/log/compute_mean.log \
    ivector-mean scp:exp/ivectors_train/ivector.scp \
    exp/ivectors_train/mean.vec || exit 1;

  # This script uses LDA to decrease the dimensionality prior to PLDA.
  lda_dim=200
  $train_cmd exp/ivectors_train/log/lda.log \
    ivector-compute-lda --total-covariance-factor=0.0 --dim=$lda_dim \
    "ark:ivector-subtract-global-mean scp:exp/ivectors_train/ivector.scp ark:- |" \
    ark:data/train/utt2spk exp/ivectors_train/transform.mat || exit 1;

  # Train the PLDA model.
  $train_cmd exp/ivectors_train/log/plda.log \
    ivector-compute-plda ark:data/train/spk2utt \
    "ark:ivector-subtract-global-mean scp:exp/ivectors_train/ivector.scp ark:- | transform-vec exp/ivectors_train/transform.mat ark:- ark:- | ivector-normalize-length ark:-  ark:- |" \
    exp/ivectors_train/plda || exit 1;
fi

if [ $stage -le 8 ]; then
  $train_cmd exp/scores/log/voxceleb1_test_scoring.log \
    ivector-plda-scoring --normalize-length=true \
    "ivector-copy-plda --smoothing=0.0 exp/ivectors_train/plda - |" \
    "ark:ivector-subtract-global-mean exp/ivectors_train/mean.vec scp:exp/ivectors_voxceleb1_test/ivector.scp ark:- | transform-vec exp/ivectors_train/transform.mat ark:- ark:- | ivector-normalize-length ark:- ark:- |" \
    "ark:ivector-subtract-global-mean exp/ivectors_train/mean.vec scp:exp/ivectors_voxceleb1_test/ivector.scp ark:- | transform-vec exp/ivectors_train/transform.mat ark:- ark:- | ivector-normalize-length ark:- ark:- |" \
    "cat '$voxceleb1_trials' | cut -d\  --fields=1,2 |" exp/scores_voxceleb1_test || exit 1;
fi

if [ $stage -le 9 ]; then
  eer=`compute-eer <(local/prepare_for_eer.py $voxceleb1_trials exp/scores_voxceleb1_test) 2> /dev/null`
  echo "EER: ${eer}%"
  # For reference, here's the ivector system from ../v1:
  # EER: 5.53%
fi
