#!/bin/bash
# Copyright   2017   Johns Hopkins University (Author: Daniel Garcia-Romero)
#             2017   Johns Hopkins University (Author: Daniel Povey)
#        2017-2018   David Snyder
#             2018   Ewald Enzinger
#
# Apache 2.0.
#
# This recipe demonstrates an approach using bottleneck features (BNF),
# full-covariance GMM-UBM, iVectors, and a PLDA backend. Two variants
# are shown below, one using BNF for calculating posterior probabilities
# and using MFCC+BNF as speaker ID features, and another using BNF for
# calculating posterior probabilities but only MFCCs as speaker ID features.
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
  utils/combine_data.sh data/train data/voxceleb2_train \
    data/voxceleb2_test data/voxceleb1_train
fi

if [ $stage -le 2 ]; then
  # Make MFCCs and compute the energy-based VAD for each dataset
  for name in train voxceleb1_test; do
    cp -r data/${name} data/${name}_hires
    steps/make_mfcc.sh --write-utt2num-frames true \
      --mfcc-config conf/mfcc.conf --nj 40 --cmd "$train_cmd" \
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
    steps/nnet3/make_bottleneck_features.sh --nj 40 --cmd "$train_cmd" \
      tdnn_bn.batchnorm data/${name}_hires data/${name}_bnf \
      exp/nnet3 exp/make_bnf $bnfdir
    cp data/${name}/vad.scp data/${name}_bnf/
    utils/fix_data_dir.sh data/${name}_bnf
  done
fi

if [ $stage -le 4 ]; then
  # Train the UBM using bottleneck features.
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
  # In this stage, we train the i-vector extractor.
  # We use bottleneck features and the UBM trained above to compute posterior
  # probabilities. We compute i-vector stats using combined MFCC+bottleneck
  # feature vectors (--add-bnf true).
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
  utils/subset_data_dir.sh --utt-list data/train_100k/utt2spk \
    data/train_bnf data/train_bnf_100k
  # Train the i-vector extractor.
  local/nnet3/bnf/train_ivector_extractor.sh --cmd "$train_cmd --mem 16G" \
    --ivector-dim 600 --delta-window 3 --delta-order 2 --add-bnf true \
    --num-iters 5 exp/full_ubm/final.ubm data/train_100k data/train_bnf_100k \
    exp/extractor_bnf
fi

if [ $stage -le 6 ]; then
  local/nnet3/bnf/extract_ivectors.sh --cmd "$train_cmd --mem 8G" --nj 80 --add-bnf true \
    exp/extractor_bnf data/train data/train_bnf \
    exp/ivectors_train

  local/nnet3/bnf/extract_ivectors.sh --cmd "$train_cmd --mem 8G" --nj 40 --add-bnf true \
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
  mindcf1=`sid/compute_min_dcf.py --p-target 0.01 exp/scores_voxceleb1_test $voxceleb1_trials 2> /dev/null`
  mindcf2=`sid/compute_min_dcf.py --p-target 0.001 exp/scores_voxceleb1_test $voxceleb1_trials 2> /dev/null`
  echo "MFCC-BNF EER: ${eer}%"
  echo "minDCF(p-target=0.01): $mindcf1"
  echo "minDCF(p-target=0.001): $mindcf2"
  # MFCC-BNF EER: 4.205%
  # minDCF(p-target=0.01): 0.4509
  # minDCF(p-target=0.001): 0.6528
  #
  # For reference, here's the ivector system from ../v1:
  # EER: 5.419%
  # minDCF(p-target=0.01): 0.4701
  # minDCF(p-target=0.001): 0.5981
fi

# Next, we evaluate another bottleneck feature system variant that uses
# bottleneck features to compute posterior probabilities, but uses only
# MFCC features to compute i-vector stats.

if [ $stage -le 10 ]; then
  # We use bottleneck features and the previously trained UBM to compute
  # posterior probabilities. We compute i-vector stats using only MFCCs.
  local/nnet3/bnf/train_ivector_extractor.sh --cmd "$train_cmd --mem 16G" \
    --ivector-dim 600 --delta-window 3 --delta-order 2 \
    --num-iters 5 exp/full_ubm/final.ubm data/train_100k data/train_bnf_100k \
    exp/extractor_bnf_onlymfcc
fi

if [ $stage -le 11 ]; then
  local/nnet3/bnf/extract_ivectors.sh --cmd "$train_cmd --mem 5G" --nj 80 \
    exp/extractor_bnf_onlymfcc data/train data/train_bnf \
    exp/ivectors_onlymfcc_train

  local/nnet3/bnf/extract_ivectors.sh --cmd "$train_cmd --mem 5G" --nj 40 \
    exp/extractor_bnf_onlymfcc data/voxceleb1_test data/voxceleb1_test_bnf \
    exp/ivectors_onlymfcc_voxceleb1_test
fi

if [ $stage -le 12 ]; then
  # Compute the mean vector for centering the evaluation i-vectors.
  $train_cmd exp/ivectors_onlymfcc_train/log/compute_mean.log \
    ivector-mean scp:exp/ivectors_onlymfcc_train/ivector.scp \
    exp/ivectors_onlymfcc_train/mean.vec || exit 1;

  # This script uses LDA to decrease the dimensionality prior to PLDA.
  lda_dim=200
  $train_cmd exp/ivectors_onlymfcc_train/log/lda.log \
    ivector-compute-lda --total-covariance-factor=0.0 --dim=$lda_dim \
    "ark:ivector-subtract-global-mean scp:exp/ivectors_onlymfcc_train/ivector.scp ark:- |" \
    ark:data/train/utt2spk exp/ivectors_onlymfcc_train/transform.mat || exit 1;

  # Train the PLDA model.
  $train_cmd exp/ivectors_onlymfcc_train/log/plda.log \
    ivector-compute-plda ark:data/train/spk2utt \
    "ark:ivector-subtract-global-mean scp:exp/ivectors_onlymfcc_train/ivector.scp ark:- | transform-vec exp/ivectors_onlymfcc_train/transform.mat ark:- ark:- | ivector-normalize-length ark:-  ark:- |" \
    exp/ivectors_onlymfcc_train/plda || exit 1;
fi

if [ $stage -le 13 ]; then
  $train_cmd exp/scores/log/voxceleb1_test_onlymfcc_scoring.log \
    ivector-plda-scoring --normalize-length=true \
    "ivector-copy-plda --smoothing=0.0 exp/ivectors_onlymfcc_train/plda - |" \
    "ark:ivector-subtract-global-mean exp/ivectors_onlymfcc_train/mean.vec scp:exp/ivectors_onlymfcc_voxceleb1_test/ivector.scp ark:- | transform-vec exp/ivectors_onlymfcc_train/transform.mat ark:- ark:- | ivector-normalize-length ark:- ark:- |" \
    "ark:ivector-subtract-global-mean exp/ivectors_onlymfcc_train/mean.vec scp:exp/ivectors_onlymfcc_voxceleb1_test/ivector.scp ark:- | transform-vec exp/ivectors_onlymfcc_train/transform.mat ark:- ark:- | ivector-normalize-length ark:- ark:- |" \
    "cat '$voxceleb1_trials' | cut -d\  --fields=1,2 |" exp/scores_onlymfcc_voxceleb1_test || exit 1;
fi

if [ $stage -le 14 ]; then
  eer=`compute-eer <(local/prepare_for_eer.py $voxceleb1_trials exp/scores_onlymfcc_voxceleb1_test) 2> /dev/null`
  mindcf1=`sid/compute_min_dcf.py --p-target 0.01 exp/scores_onlymfcc_voxceleb1_test $voxceleb1_trials 2> /dev/null`
  mindcf2=`sid/compute_min_dcf.py --p-target 0.001 exp/scores_onlymfcc_voxceleb1_test $voxceleb1_trials 2> /dev/null`
  echo "only MFCC EER: ${eer}%"
  echo "minDCF(p-target=0.01): $mindcf1"
  echo "minDCF(p-target=0.001): $mindcf2"
  # only MFCC EER: 4.608%
  # minDCF(p-target=0.01): 0.4370
  # minDCF(p-target=0.001): 0.6094
  #
  # For reference, here's the ivector system from ../v1:
  # EER: 5.419%
  # minDCF(p-target=0.01): 0.4701
  # minDCF(p-target=0.001): 0.5981
fi

if [ $stage -le 15 ]; then
  local/nnet3/bnf/train_ivector_extractor_bnf.sh --cmd "$train_cmd --mem 18G" \
    --ivector-dim 600 --delta-window 3 --delta-order 1 \
    --num-iters 5 exp/full_ubm/final.ubm data/train_bnf_100k data/train_bnf_100k \
    exp/extractor_bnf_onlybnf
fi

if [ $stage -le 16 ]; then
  local/nnet3/bnf/extract_ivectors_bnf.sh --cmd "$train_cmd --mem 7G" --nj 80 \
    exp/extractor_bnf_onlybnf data/train_bnf data/train_bnf \
    exp/ivectors_onlybnf_train

  local/nnet3/bnf/extract_ivectors_bnf.sh --cmd "$train_cmd --mem 7G" --nj 40 \
    exp/extractor_bnf_onlybnf data/voxceleb1_test_bnf data/voxceleb1_test_bnf \
    exp/ivectors_onlybnf_voxceleb1_test
fi

if [ $stage -le 17 ]; then
  # Compute the mean vector for centering the evaluation i-vectors.
  $train_cmd exp/ivectors_onlybnf_train/log/compute_mean.log \
    ivector-mean scp:exp/ivectors_onlybnf_train/ivector.scp \
    exp/ivectors_onlybnf_train/mean.vec || exit 1;

  # This script uses LDA to decrease the dimensionality prior to PLDA.
  lda_dim=200
  $train_cmd exp/ivectors_onlybnf_train/log/lda.log \
    ivector-compute-lda --total-covariance-factor=0.0 --dim=$lda_dim \
    "ark:ivector-subtract-global-mean scp:exp/ivectors_onlybnf_train/ivector.scp ark:- |" \
    ark:data/train_bnf/utt2spk exp/ivectors_onlybnf_train/transform.mat || exit 1;

  # Train the PLDA model.
  $train_cmd exp/ivectors_onlybnf_train/log/plda.log \
    ivector-compute-plda ark:data/train_bnf/spk2utt \
    "ark:ivector-subtract-global-mean scp:exp/ivectors_onlybnf_train/ivector.scp ark:- | transform-vec exp/ivectors_onlybnf_train/transform.mat ark:- ark:- | ivector-normalize-length ark:-  ark:- |" \
    exp/ivectors_onlybnf_train/plda || exit 1;
fi

if [ $stage -le 18 ]; then
  $train_cmd exp/scores/log/voxceleb1_test_onlybnf_scoring.log \
    ivector-plda-scoring --normalize-length=true \
    "ivector-copy-plda --smoothing=0.0 exp/ivectors_onlybnf_train/plda - |" \
    "ark:ivector-subtract-global-mean exp/ivectors_onlybnf_train/mean.vec scp:exp/ivectors_onlybnf_voxceleb1_test/ivector.scp ark:- | transform-vec exp/ivectors_onlybnf_train/transform.mat ark:- ark:- | ivector-normalize-length ark:- ark:- |" \
    "ark:ivector-subtract-global-mean exp/ivectors_onlybnf_train/mean.vec scp:exp/ivectors_onlybnf_voxceleb1_test/ivector.scp ark:- | transform-vec exp/ivectors_onlybnf_train/transform.mat ark:- ark:- | ivector-normalize-length ark:- ark:- |" \
    "cat '$voxceleb1_trials' | cut -d\  --fields=1,2 |" exp/scores_onlybnf_voxceleb1_test || exit 1;
fi

if [ $stage -le 19 ]; then
  eer=`compute-eer <(local/prepare_for_eer.py $voxceleb1_trials exp/scores_onlybnf_voxceleb1_test) 2> /dev/null`
  mindcf1=`sid/compute_min_dcf.py --p-target 0.01 exp/scores_onlybnf_voxceleb1_test $voxceleb1_trials 2> /dev/null`
  mindcf2=`sid/compute_min_dcf.py --p-target 0.001 exp/scores_onlybnf_voxceleb1_test $voxceleb1_trials 2> /dev/null`
  echo "only BNF EER: ${eer}%"
  echo "minDCF(p-target=0.01): $mindcf1"
  echo "minDCF(p-target=0.001): $mindcf2"
  # only BNF EER:
  # minDCF(p-target=0.01):
  # minDCF(p-target=0.001):
fi

if [ $stage -le 20 ]; then
  # Perform i-vector level fusion
  for name in train voxceleb1_test; do
    mkdir -p exp/ivectors_combined_${name}
    $train_cmd exp/ivectors_combined_${name}/log/combine_ivectors.log \
      paste-ivectors --print-args=false scp:exp/ivectors_${name}/ivector.scp \
      scp:exp/ivectors_onlybnf_${name}/ivector.scp \
      ark,scp:exp/ivectors_combined_${name}/ivector.ark,exp/ivectors_combined_${name}/ivector.scp
  done

  # Compute the mean vector for centering the evaluation i-vectors.
  $train_cmd exp/ivectors_combined_train/log/compute_mean.log \
    ivector-mean scp:exp/ivectors_combined_train/ivector.scp \
    exp/ivectors_combined_train/mean.vec || exit 1;

  # This script uses LDA to decrease the dimensionality prior to PLDA.
  lda_dim=400
  $train_cmd exp/ivectors_combined_train/log/lda.log \
    ivector-compute-lda --total-covariance-factor=0.0 --dim=$lda_dim \
    "ark:ivector-subtract-global-mean scp:exp/ivectors_combined_train/ivector.scp ark:- |" \
    ark:data/train/utt2spk exp/ivectors_combined_train/transform.mat || exit 1;

  # Train the PLDA model.
  $train_cmd exp/ivectors_combined_train/log/plda.log \
    ivector-compute-plda ark:data/train/spk2utt \
    "ark:ivector-subtract-global-mean scp:exp/ivectors_combined_train/ivector.scp ark:- | transform-vec exp/ivectors_combined_train/transform.mat ark:- ark:- | ivector-normalize-length ark:-  ark:- |" \
    exp/ivectors_combined_train/plda || exit 1;
fi

if [ $stage -le 21 ]; then
  $train_cmd exp/scores/log/voxceleb1_test_combined_scoring.log \
    ivector-plda-scoring --normalize-length=true \
    "ivector-copy-plda --smoothing=0.0 exp/ivectors_combined_train/plda - |" \
    "ark:ivector-subtract-global-mean exp/ivectors_combined_train/mean.vec scp:exp/ivectors_combined_voxceleb1_test/ivector.scp ark:- | transform-vec exp/ivectors_combined_train/transform.mat ark:- ark:- | ivector-normalize-length ark:- ark:- |" \
    "ark:ivector-subtract-global-mean exp/ivectors_combined_train/mean.vec scp:exp/ivectors_combined_voxceleb1_test/ivector.scp ark:- | transform-vec exp/ivectors_combined_train/transform.mat ark:- ark:- | ivector-normalize-length ark:- ark:- |" \
    "cat '$voxceleb1_trials' | cut -d\  --fields=1,2 |" exp/scores_combined_voxceleb1_test || exit 1;
fi

if [ $stage -le 22 ]; then
  eer=`compute-eer <(local/prepare_for_eer.py $voxceleb1_trials exp/scores_combined_voxceleb1_test) 2> /dev/null`
  mindcf1=`sid/compute_min_dcf.py --p-target 0.01 exp/scores_combined_voxceleb1_test $voxceleb1_trials 2> /dev/null`
  mindcf2=`sid/compute_min_dcf.py --p-target 0.001 exp/scores_combined_voxceleb1_test $voxceleb1_trials 2> /dev/null`
  echo "i-vector fusion EER: ${eer}%"
  echo "minDCF(p-target=0.01): $mindcf1"
  echo "minDCF(p-target=0.001): $mindcf2"
  # only BNF EER:
  # minDCF(p-target=0.01):
  # minDCF(p-target=0.001):
fi
