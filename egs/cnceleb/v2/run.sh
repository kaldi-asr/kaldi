#!/bin/bash
# Copyright    2017   Johns Hopkins University (Author: Daniel Povey)
#              2017   Johns Hopkins University (Author: Daniel Garcia-Romero)
#              2018   Ewald Enzinger
#              2018   David Snyder
#              2019   Tsinghua University (Author: Jiawen Kang and Lantian Li)
# Apache 2.0.
#
# This is an x-vector-based recipe for CN-Celeb database.
# It is based on "X-vectors: Robust DNN Embeddings for Speaker Recognition"
# by Snyder et al. The recipe uses CN-Celeb/dev for training the UBM, T matrix 
# and PLDA, and CN-Celeb/eval for evaluation. The results are reported in terms 
# of EER and minDCF, and are inline in the comments below.

. ./cmd.sh
. ./path.sh
set -e
mfccdir=`pwd`/_mfcc
vaddir=`pwd`/_vad

cnceleb_root=/export/corpora/CN-Celeb
nnet_dir=exp/xvector_nnet_1a
eval_trials_core=data/eval_test/trials/trials.lst

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

if [ $stage -le 3 ]; then
  # Note that there are over one-third of the utterances less than 2 seconds in our training set,
  # and these short utterances are harmful for DNNs x-vector training. Therefore, to improve 
  # performance of DNN training, we will combine the short utterances longer than 5 seconds.
  local/combine_short_segments.sh data/train 5 data/train_comb
  # Compute the energy-based VAD for train_comb
  sid/compute_vad_decision.sh --nj 20 --cmd "$train_cmd" \
    data/train_comb exp/make_vad $vaddir
  utils/fix_data_dir.sh data/train_comb
fi

# Now we prepare the features to generate examples for xvector training.
if [ $stage -le 4 ]; then
  # This script applies CMVN and removes nonspeech frames. Note that this is somewhat
  # wasteful, as it roughly doubles the amount of training data on disk. After
  # creating training examples, this can be removed.
  local/nnet3/xvector/prepare_feats_for_egs.sh --nj 20 --cmd "$train_cmd" \
    data/train_comb data/train_comb_no_sil exp/train_comb_no_sil
  utils/fix_data_dir.sh data/train_comb_no_sil
fi

if [ $stage -le 5 ]; then
  # Now, we need to remove features that are too short after removing silence
  # frames. We want atleast 5s (500 frames) per utterance.
  min_len=400
  mv data/train_comb_no_sil/utt2num_frames data/train_comb_no_sil/utt2num_frames.bak
  awk -v min_len=${min_len} '$2 > min_len {print $1, $2}' data/train_comb_no_sil/utt2num_frames.bak > data/train_comb_no_sil/utt2num_frames
  utils/filter_scp.pl data/train_comb_no_sil/utt2num_frames data/train_comb_no_sil/utt2spk > data/train_comb_no_sil/utt2spk.new
  mv data/train_comb_no_sil/utt2spk.new data/train_comb_no_sil/utt2spk
  utils/fix_data_dir.sh data/train_comb_no_sil

  # We also want several utterances per speaker. Now we'll throw out speakers
  # with fewer than 8 utterances.
  min_num_utts=8
  awk '{print $1, NF-1}' data/train_comb_no_sil/spk2utt > data/train_comb_no_sil/spk2num
  awk -v min_num_utts=${min_num_utts} '$2 >= min_num_utts {print $1, $2}' data/train_comb_no_sil/spk2num | utils/filter_scp.pl - data/train_comb_no_sil/spk2utt > data/train_comb_no_sil/spk2utt.new
  mv data/train_comb_no_sil/spk2utt.new data/train_comb_no_sil/spk2utt
  utils/spk2utt_to_utt2spk.pl data/train_comb_no_sil/spk2utt > data/train_comb_no_sil/utt2spk

  utils/filter_scp.pl data/train_comb_no_sil/utt2spk data/train_comb_no_sil/utt2num_frames > data/train_comb_no_sil/utt2num_frames.new
  mv data/train_comb_no_sil/utt2num_frames.new data/train_comb_no_sil/utt2num_frames

  # Now we're ready to create training examples.
  utils/fix_data_dir.sh data/train_comb_no_sil
fi
  
# Stages 6 through 8 are handled in run_xvector.sh
local/nnet3/xvector/run_xvector.sh --stage $stage --train-stage -1 \
  --data data/train_comb_no_sil --nnet-dir $nnet_dir \
  --egs-dir $nnet_dir/egs

if [ $stage -le 9 ]; then
  # These x-vectors will be used for mean-subtraction, LDA, and PLDA training.
  sid/nnet3/xvector/extract_xvectors.sh --cmd "$train_cmd --mem 4G" --nj 20 \
    $nnet_dir data/train_comb \
    $nnet_dir/xvectors_train_comb

  # Extract x-vector for eval sets.
  for name in eval_enroll eval_test; do
    sid/nnet3/xvector/extract_xvectors.sh --cmd "$train_cmd --mem 4G" --nj 20 \
      $nnet_dir data/$name \
      $nnet_dir/xvectors_$name
  done
fi

if [ $stage -le 10 ]; then
  # Compute the mean.vec used for centering.
  $train_cmd $nnet_dir/xvectors_train_comb/log/compute_mean.log \
    ivector-mean scp:$nnet_dir/xvectors_train_comb/xvector.scp \
    $nnet_dir/xvectors_train_comb/mean.vec || exit 1;

  # Use LDA to decrease the dimensionality prior to PLDA.
  lda_dim=128
  $train_cmd $nnet_dir/xvectors_train_comb/log/lda.log \
    ivector-compute-lda --total-covariance-factor=0.0 --dim=$lda_dim \
    "ark:ivector-subtract-global-mean scp:$nnet_dir/xvectors_train_comb/xvector.scp ark:- |" \
    ark:data/train_comb/utt2spk $nnet_dir/xvectors_train_comb/transform.mat || exit 1;

  # Train the PLDA model.
  $train_cmd $nnet_dir/xvectors_train_comb/log/plda.log \
    ivector-compute-plda ark:data/train_comb/spk2utt \
    "ark:ivector-subtract-global-mean scp:$nnet_dir/xvectors_train_comb/xvector.scp ark:- | transform-vec $nnet_dir/xvectors_train_comb/transform.mat ark:- ark:- | ivector-normalize-length ark:- ark:- |" \
    $nnet_dir/xvectors_train_comb/plda || exit 1;
fi

if [ $stage -le 11 ]; then
  # Compute PLDA scores for CN-Celeb eval core trials
  $train_cmd $nnet_dir/scores/log/cnceleb_eval_scoring.log \
    ivector-plda-scoring --normalize-length=true \
    --num-utts=ark:$nnet_dir/xvectors_eval_enroll/num_utts.ark \
    "ivector-copy-plda --smoothing=0.0 $nnet_dir/xvectors_train_comb/plda - |" \
    "ark:ivector-mean ark:data/eval_enroll/spk2utt scp:$nnet_dir/xvectors_eval_enroll/xvector.scp ark:- | ivector-subtract-global-mean $nnet_dir/xvectors_train_comb/mean.vec ark:- ark:- | transform-vec $nnet_dir/xvectors_train_comb/transform.mat ark:- ark:- | ivector-normalize-length ark:- ark:- |" \
    "ark:ivector-subtract-global-mean $nnet_dir/xvectors_train_comb/mean.vec scp:$nnet_dir/xvectors_eval_test/xvector.scp ark:- | transform-vec $nnet_dir/xvectors_train_comb/transform.mat ark:- ark:- | ivector-normalize-length ark:- ark:- |" \
    "cat '$eval_trials_core' | cut -d\  --fields=1,2 |" $nnet_dir/scores/cnceleb_eval_scores || exit 1;

  # CN-Celeb Eval Core:
  # EER: 14.70%
  # minDCF(p-target=0.01): 0.6814
  # minDCF(p-target=0.001): 0.7979
  echo -e "\nCN-Celeb Eval Core:";
  eer=$(paste $eval_trials_core $nnet_dir/scores/cnceleb_eval_scores | awk '{print $6, $3}' | compute-eer - 2>/dev/null)
  mindcf1=`sid/compute_min_dcf.py --p-target 0.01 $nnet_dir/scores/cnceleb_eval_scores $eval_trials_core 2>/dev/null`
  mindcf2=`sid/compute_min_dcf.py --p-target 0.001 $nnet_dir/scores/cnceleb_eval_scores $eval_trials_core 2>/dev/null`
  echo "EER: $eer%"
  echo "minDCF(p-target=0.01): $mindcf1"
  echo "minDCF(p-target=0.001): $mindcf2"
fi
