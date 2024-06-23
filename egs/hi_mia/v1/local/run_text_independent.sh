#!/usr/bin/env bash
# Copyright 2020 Xuechen LIU
# Apache 2.0.
#
# perform text-independent training & scoring on HIMIA
set -e
. ./path.sh
. ./cmd.sh

stage=8
lda_dim=200
nj=10
use_gpu=false
. ./utils/parse_options.sh

nnetdir=$1
sc_trial=$2
mc_trial=$3

# xvector embedding extraction
if [ $stage -le 8 ]; then
    for set in test dev train; do
        sid/nnet3/xvector/extract_xvectors.sh \
            --cmd "$train_cmd --mem 4G" --nj $nj --use-gpu $use_gpu \
            $nnetdir data/himia/$set $nnetdir/xvectors_himia_$set
    done
fi

if [ $stage -le 9 ]; then
  # Compute the mean.vec used for centering.
  $train_cmd $nnetdir/xvectors_himia_train/log/compute_mean.log \
      ivector-mean scp:$nnetdir/xvectors_himia_train/xvector.scp \
      $nnetdir/xvectors_himia_train/mean.vec || exit 1;

  # Use LDA to decrease the dimensionality prior to PLDA.
  $train_cmd $nnetdir/xvectors_himia_train/log/lda.log \
    ivector-compute-lda --total-covariance-factor=0.0 --dim=$lda_dim \
    "ark:ivector-subtract-global-mean scp:$nnetdir/xvectors_himia_train/xvector.scp ark:- |" \
    ark:data/himia/train/utt2spk $nnetdir/xvectors_himia_train/transform.mat || exit 1;

  # Train the PLDA model.
  $train_cmd $nnetdir/xvectors_himia_train/log/plda.log \
    ivector-compute-plda ark:data/himia/train/spk2utt \
    "ark:ivector-subtract-global-mean scp:$nnetdir/xvectors_himia_train/xvector.scp ark:- | transform-vec $nnetdir/xvectors_himia_train/transform.mat ark:- ark:- | ivector-normalize-length ark:-  ark:- |" \
    $nnetdir/xvectors_himia_train/plda || exit 1;
fi

if [ $stage -le 11 ]; then
  # Compute PLDA scores for clean mic trial
  $train_cmd $nnetdir/scores/log/himia_sc_scoring.log \
    ivector-plda-scoring --normalize-length=true \
    "ivector-copy-plda --smoothing=0.0 $nnetdir/xvectors_himia_train/plda - |" \
    "ark:ivector-mean ark:data/himia/test/spk2utt scp:$nnetdir/xvectors_himia_test/xvector.scp ark:- | ivector-subtract-global-mean $nnetdir/xvectors_himia_train/mean.vec ark:- ark:- | transform-vec $nnetdir/xvectors_himia_train/transform.mat ark:- ark:- | ivector-normalize-length ark:- ark:- |" \
    "ark:ivector-mean ark:data/himia/test/spk2utt scp:$nnetdir/xvectors_himia_test/xvector.scp ark:- | ivector-subtract-global-mean $nnetdir/xvectors_himia_train/mean.vec ark:- ark:- | transform-vec $nnetdir/xvectors_himia_train/transform.mat ark:- ark:- | ivector-normalize-length ark:- ark:- |" \
    "cat '$sc_trial' | cut -d\  --fields=1,2 |" $nnetdir/scores/test_sc_scores || exit 1;

  # HIMIA far-field microphone array single-channel result (no AISHELL2 for xvector training):                                                                                                                                              
  # EER: 8.086%                                                                                                                                                                                          
  # minDCF(p-target=0.01): 0.6328                                                                                                                                                                        
  # minDCF(p-target=0.001): 0.8407                                                                                                                                                                       
  # HIMIA far-field microphone array single-channel result (with AISHELL2 for xvector training):
  # EER: 6.691%
  # minDCF(p-target=0.01): 0.5161
  # minDCF(p-target=0.001): 0.6339
  echo -e "\nHIMIA far-field microphone array single-channel result:";
  eer=$(paste $sc_trial $nnetdir/scores/test_sc_scores | awk '{print $6, $3}' | compute-eer - 2>/dev/null)
  mindcf1=`sid/compute_min_dcf.py --p-target 0.01 $nnetdir/scores/test_sc_scores $sc_trial 2> /dev/null`
  mindcf2=`sid/compute_min_dcf.py --p-target 0.001 $nnetdir/scores/test_sc_scores $sc_trial 2> /dev/null`
  echo "EER: $eer%"
  echo "minDCF(p-target=0.01): $mindcf1"
  echo "minDCF(p-target=0.001): $mindcf2"
fi

if [ $stage -le 12 ]; then
  # Compute PLDA scores for clean mic trial
  $train_cmd $nnetdir/scores/log/himia_mc_scoring.log \
    ivector-plda-scoring --normalize-length=true \
    "ivector-copy-plda --smoothing=0.0 $nnetdir/xvectors_himia_train/plda - |" \
    "ark:ivector-mean ark:data/himia/test/spk2utt scp:$nnetdir/xvectors_himia_test/xvector.scp ark:- | ivector-subtract-global-mean $nnetdir/xvectors_himia_train/mean.vec ark:- ark:- | transform-vec $nnetdir/xvectors_himia_train/transform.mat ark:- ark:- | ivector-normalize-length ark:- ark:- |" \
    "ark:ivector-mean ark:data/himia/test/spk2utt scp:$nnetdir/xvectors_himia_test/xvector.scp ark:- | ivector-subtract-global-mean $nnetdir/xvectors_himia_train/mean.vec ark:- ark:- | transform-vec $nnetdir/xvectors_himia_train/transform.mat ark:- ark:- | ivector-normalize-length ark:- ark:- |" \
    "cat '$mc_trial' | cut -d\  --fields=1,2 |" $nnetdir/scores/test_mc_scores || exit 1;

  # HIMIA far-field microphone array multi-channel result (no AISHELL2 for xvector training):                                                                                                                                               
  # EER: 9.742%                                                                                                                                                                                          
  # minDCF(p-target=0.01): 0.7512                                                                                                                                                                        
  # minDCF(p-target=0.001): 0.8388 
  # HIMIA far-field microphone array multi-channel result (with AISHELL2 for xvector training):
  # EER: 8.573%
  # minDCF(p-target=0.01): 0.6963
  # minDCF(p-target=0.001): 0.8732
  echo -e "\nHIMIA far-field microphone array multi-channel result:";
  eer=$(paste $mc_trial $nnetdir/scores/test_mc_scores | awk '{print $6, $3}' | compute-eer - 2>/dev/null)
  mindcf1=`sid/compute_min_dcf.py --p-target 0.01 $nnetdir/scores/test_mc_scores $mc_trial 2> /dev/null`
  mindcf2=`sid/compute_min_dcf.py --p-target 0.001 $nnetdir/scores/test_mc_scores $mc_trial 2> /dev/null`
  echo "EER: $eer%"
  echo "minDCF(p-target=0.01): $mindcf1"
  echo "minDCF(p-target=0.001): $mindcf2"
fi

wait;
exit 0;
