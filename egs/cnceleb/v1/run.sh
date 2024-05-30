#!/usr/bin/env bash
# Copyright   2017   Johns Hopkins University (Author: Daniel Garcia-Romero)
#             2017   Johns Hopkins University (Author: Daniel Povey)
#        2017-2018   David Snyder
#             2018   Ewald Enzinger
#             2020   Tsinghua University (Author: Ruiqi Liu and Lantian Li)
# Apache 2.0.
#
# This is an i-vector-based recipe for CN-Celeb1 and CN-Celeb2 database.
# The recipe uses CN-Celeb1/dev and CN-Celeb2 for training the UBM, T matrix, PLDA;
# and CN-Celeb1/eval for evaluation. The results are reported in terms of 
# EER and minDCF, and are inline in the comments below.

. ./cmd.sh
. ./path.sh
set -e
mfccdir=`pwd`/mfcc
vaddir=`pwd`/mfcc

cnceleb1_root=/export/corpora/CN-Celeb1
cnceleb2_root=/export/corpora/CN-Celeb2
eval_trails_core=data/eval_test/trials/trials.lst

stage=0

if [ $stage -le 0 ]; then
  # This script creates data/cnceleb1_train and data/eval_{enroll,test}.
  # Our evaluation set is the eval portion of CN-Celeb1.
  local/make_cnceleb1.sh $cnceleb1_root data

  # This script creates data/cnceleb2_train of CN-Celeb2.
  local/make_cnceleb2.sh $cnceleb2_root data/cnceleb2_train

  # We'll train on all of CN-Celeb2, plus the training portion of CN-Celeb1.
  # This should give 2,800 speakers and 640,744 utterances.
  utils/combine_data.sh data/train data/cnceleb1_train data/cnceleb2_train
fi

if [ $stage -le 1 ]; then
  # Make MFCCs and compute the energy-based VAD for each dataset
  for name in train eval_enroll eval_test; do
    steps/make_mfcc.sh --write-utt2num-frames true \
      --mfcc-config conf/mfcc.conf --nj 40 --cmd "$train_cmd" \
      data/${name} exp/make_mfcc $mfccdir
    utils/fix_data_dir.sh data/${name}
    sid/compute_vad_decision.sh --nj 40 --cmd "$train_cmd" \
      data/${name} exp/make_vad $vaddir
    utils/fix_data_dir.sh data/${name}
  done
fi

if [ $stage -le 2 ]; then
  # Train the UBM on CN-Celeb 1 and 2.
  sid/train_diag_ubm.sh --cmd "$train_cmd --mem 4G " \
    --nj 40 --num-threads 1 \
    data/train 2048 \
    exp/diag_ubm

  sid/train_full_ubm.sh --cmd "$train_cmd" \
    --nj 40 --remove-low-count-gaussians false \
    data/train \
    exp/diag_ubm exp/full_ubm
fi

if [ $stage -le 3 ]; then
  # In this stage, we train the i-vector extractor.
  #
  # Note that there are well over 600k utterances in our training set,
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

if [ $stage -le 4 ]; then
  # These i-vectors will be used for mean-subtraction, LDA, and PLDA training.
  sid/extract_ivectors.sh --cmd "$train_cmd --mem 4G" --nj 80 \
    exp/extractor data/train_100k \
    exp/ivectors_train_100k

  # Extract i-vector for eval sets.
  for name in eval_enroll eval_test; do
    sid/extract_ivectors.sh --cmd "$train_cmd --mem 4G" --nj 40 \
      exp/extractor data/$name \
      exp/ivectors_$name
  done
fi

if [ $stage -le 5 ]; then
  # Compute the mean vector for centering the evaluation i-vectors.
  $train_cmd exp/ivectors_train_100k/log/compute_mean.log \
    ivector-mean scp:exp/ivectors_train_100k/ivector.scp \
    exp/ivectors_train_100k/mean.vec || exit 1;

  # This script uses LDA to decrease the dimensionality prior to PLDA.
  lda_dim=150
  $train_cmd exp/ivectors_train_100k/log/lda.log \
    ivector-compute-lda --total-covariance-factor=0.0 --dim=$lda_dim \
    "ark:ivector-subtract-global-mean scp:exp/ivectors_train_100k/ivector.scp ark:- |" \
    ark:data/train_100k/utt2spk exp/ivectors_train_100k/transform.mat || exit 1;

  # Train the PLDA model.
  $train_cmd exp/ivectors_train_100k/log/plda.log \
    ivector-compute-plda ark:data/train_100k/spk2utt \
    "ark:ivector-subtract-global-mean scp:exp/ivectors_train_100k/ivector.scp ark:- | transform-vec exp/ivectors_train_100k/transform.mat ark:- ark:- | ivector-normalize-length ark:- ark:- |" \
    exp/ivectors_train_100k/plda || exit 1;
fi

if [ $stage -le 6 ]; then
  # Compute PLDA scores for CN-Celeb eval core trials
  $train_cmd exp/scores/log/cnceleb_eval_scoring.log \
    ivector-plda-scoring --normalize-length=true \
    --num-utts=ark:exp/ivectors_eval_enroll/num_utts.ark \
    "ivector-copy-plda --smoothing=0.0 exp/ivectors_train_100k/plda - |" \
    "ark:ivector-mean ark:data/eval_enroll/spk2utt scp:exp/ivectors_eval_enroll/ivector.scp ark:- | ivector-subtract-global-mean exp/ivectors_train_100k/mean.vec ark:- ark:- | transform-vec exp/ivectors_train_100k/transform.mat ark:- ark:- | ivector-normalize-length ark:- ark:- |" \
    "ark:ivector-subtract-global-mean exp/ivectors_train_100k/mean.vec scp:exp/ivectors_eval_test/ivector.scp ark:- | transform-vec exp/ivectors_train_100k/transform.mat ark:- ark:- | ivector-normalize-length ark:- ark:- |" \
    "cat '$eval_trails_core' | cut -d\  --fields=1,2 |" exp/scores/cnceleb_eval_scores || exit 1;
fi

if [ $stage -le 7 ]; then
  echo -e "\nCN-Celeb Eval Core:";
  eer=$(paste $eval_trails_core exp/scores/cnceleb_eval_scores | awk '{print $6, $3}' | compute-eer - 2>/dev/null)
  mindcf1=`sid/compute_min_dcf.py --p-target 0.01 exp/scores/cnceleb_eval_scores $eval_trails_core 2> /dev/null`
  mindcf2=`sid/compute_min_dcf.py --p-target 0.001 exp/scores/cnceleb_eval_scores $eval_trails_core 2> /dev/null`
  echo "EER: $eer%"
  echo "minDCF(p-target=0.01): $mindcf1"
  echo "minDCF(p-target=0.001): $mindcf2"
  # EER: 14.18%
  # minDCF(p-target=0.01): 0.6407
  # minDCF(p-target=0.001): 0.7458
fi
