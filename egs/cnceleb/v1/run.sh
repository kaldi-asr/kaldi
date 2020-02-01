#!/usr/bin/env bash
# Copyright   2017   Johns Hopkins University (Author: Daniel Garcia-Romero)
#             2017   Johns Hopkins University (Author: Daniel Povey)
#        2017-2018   David Snyder
#             2018   Ewald Enzinger
#             2019   Tsinghua University (Author: Jiawen Kang and Lantian Li)
# Apache 2.0.
#
# This is an i-vector-based recipe for CN-Celeb database.
# See ../README.txt for more info on data required. The recipe uses
# CN-Celeb/dev for training the UBM, T matrix and PLDA, and CN-Celeb/eval
# for evaluation. The results are reported in terms of EER and minDCF,
# and are inline in the comments below.

. ./cmd.sh
. ./path.sh
set -e
mfccdir=`pwd`/mfcc
vaddir=`pwd`/mfcc

cnceleb_root=/export/corpora/CN-Celeb
eval_trails_core=data/eval_test/trials/trials.lst

stage=0

if [ $stage -le 0 ]; then
  # Prepare the CN-Celeb dataset. The script is used to prepare the development
  # dataset and evaluation dataset.
  local/make_cnceleb.sh $cnceleb_root data
fi

if [ $stage -le 1 ]; then
  # Make MFCCs and compute the energy-based VAD for each dataset
  for name in train eval_enroll eval_test; do
    steps/make_mfcc.sh --write-utt2num-frames true --mfcc-config conf/mfcc.conf --nj 20 --cmd "$train_cmd" \
      data/${name} exp/make_mfcc $mfccdir
    utils/fix_data_dir.sh data/${name}
    sid/compute_vad_decision.sh --nj 20 --cmd "$train_cmd" \
      data/${name} exp/make_vad $vaddir
    utils/fix_data_dir.sh data/${name}
  done
fi

if [ $stage -le 2 ]; then
  # Train the UBM
  sid/train_diag_ubm.sh --cmd "$train_cmd --mem 4G" \
    --nj 20 --num-threads 8 \
    data/train 2048 \
    exp/diag_ubm

  sid/train_full_ubm.sh --cmd "$train_cmd --mem 16G" \
    --nj 20 --remove-low-count-gaussians false \
    data/train \
    exp/diag_ubm exp/full_ubm
fi

if [ $stage -le 3 ]; then
  # Train the i-vector extractor.
  sid/train_ivector_extractor.sh --nj 20 --cmd "$train_cmd --mem 16G" \
    --ivector-dim 400 --num-iters 5 \
    exp/full_ubm/final.ubm data/train \
    exp/extractor
fi

if [ $stage -le 4 ]; then
  # Note that there are over one-third of the utterances less than 2 seconds in our training set,
  # and these short utterances are harmful for PLDA training. Therefore, to improve performance 
  # of PLDA modeling and inference, we will combine the short utterances longer than 5 seconds.
  utils/data/combine_short_segments.sh --speaker-only true \
    data/train 5 data/train_comb
  # Compute the energy-based VAD for train_comb
  sid/compute_vad_decision.sh --nj 20 --cmd "$train_cmd" \
    data/train_comb exp/make_vad $vaddir
  utils/fix_data_dir.sh data/train_comb
fi

if [ $stage -le 5 ]; then
  # These i-vectors will be used for mean-subtraction, LDA, and PLDA training.
  sid/extract_ivectors.sh --cmd "$train_cmd --mem 4G" --nj 20 \
    exp/extractor data/train_comb \
    exp/ivectors_train_comb

  # Extract i-vector for eval sets.
  for name in eval_enroll eval_test; do
    sid/extract_ivectors.sh --cmd "$train_cmd --mem 4G" --nj 10 \
      exp/extractor data/$name \
      exp/ivectors_$name
  done
fi

if [ $stage -le 6 ]; then
  # Compute the mean vector for centering the evaluation i-vectors.
  $train_cmd exp/ivectors_train_comb/log/compute_mean.log \
    ivector-mean scp:exp/ivectors_train_comb/ivector.scp \
    exp/ivectors_train_comb/mean.vec || exit 1;

  # This script uses LDA to decrease the dimensionality prior to PLDA.
  lda_dim=150
  $train_cmd exp/ivectors_train_comb/log/lda.log \
    ivector-compute-lda --total-covariance-factor=0.0 --dim=$lda_dim \
    "ark:ivector-subtract-global-mean scp:exp/ivectors_train_comb/ivector.scp ark:- |" \
    ark:data/train_comb/utt2spk exp/ivectors_train_comb/transform.mat || exit 1;

  # Train the PLDA model.
  $train_cmd exp/ivectors_train_comb/log/plda.log \
    ivector-compute-plda ark:data/train_comb/spk2utt \
    "ark:ivector-subtract-global-mean scp:exp/ivectors_train_comb/ivector.scp ark:- | transform-vec exp/ivectors_train_comb/transform.mat ark:- ark:- | ivector-normalize-length ark:- ark:- |" \
    exp/ivectors_train_comb/plda || exit 1;

fi

if [ $stage -le 7 ]; then
  # Compute PLDA scores for CN-Celeb eval core trials
  $train_cmd exp/scores/log/cnceleb_eval_scoring.log \
    ivector-plda-scoring --normalize-length=true \
    --num-utts=ark:exp/ivectors_eval_enroll/num_utts.ark \
    "ivector-copy-plda --smoothing=0.0 exp/ivectors_train_comb/plda - |" \
    "ark:ivector-mean ark:data/eval_enroll/spk2utt scp:exp/ivectors_eval_enroll/ivector.scp ark:- | ivector-subtract-global-mean exp/ivectors_train_comb/mean.vec ark:- ark:- | transform-vec exp/ivectors_train_comb/transform.mat ark:- ark:- | ivector-normalize-length ark:- ark:- |" \
    "ark:ivector-subtract-global-mean exp/ivectors_train_comb/mean.vec scp:exp/ivectors_eval_test/ivector.scp ark:- | transform-vec exp/ivectors_train_comb/transform.mat ark:- ark:- | ivector-normalize-length ark:- ark:- |" \
    "cat '$eval_trails_core' | cut -d\  --fields=1,2 |" exp/scores/cnceleb_eval_scores || exit 1;

  # CN-Celeb Eval Core:
  # EER: 13.91%
  # minDCF(p-target=0.01): 0.6530
  # minDCF(p-target=0.001): 0.7521
  echo -e "\nCN-Celeb Eval Core:";
  eer=$(paste $eval_trails_core exp/scores/cnceleb_eval_scores | awk '{print $6, $3}' | compute-eer - 2>/dev/null)
  mindcf1=`sid/compute_min_dcf.py --p-target 0.01 exp/scores/cnceleb_eval_scores $eval_trails_core 2> /dev/null`
  mindcf2=`sid/compute_min_dcf.py --p-target 0.001 exp/scores/cnceleb_eval_scores $eval_trails_core 2> /dev/null`
  echo "EER: $eer%"
  echo "minDCF(p-target=0.01): $mindcf1"
  echo "minDCF(p-target=0.001): $mindcf2"
fi
