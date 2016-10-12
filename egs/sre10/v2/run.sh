#!/bin/bash
# Copyright 2015-2016   David Snyder
#                2015   Johns Hopkins University (Author: Daniel Garcia-Romero)
#                2015   Johns Hopkins University (Author: Daniel Povey)
# Apache 2.0.
#
# See README.txt for more info on data required.
# Results (EERs) are inline in comments below.
#
# This example script shows how to replace the GMM-UBM
# with a DNN trained for ASR. It also demonstrates the
# using the DNN to create a supervised-GMM.

. cmd.sh
. path.sh
set -e
mfccdir=`pwd`/mfcc
vaddir=`pwd`/mfcc
trials_female=data/sre10_test_female/trials
trials_male=data/sre10_test_male/trials
trials=data/sre10_test/trials
nnet=exp/nnet2_online/nnet_ms_a/final.mdl

# Train a DNN on about 1800 hours of the english portion of Fisher.
local/dnn/train_dnn.sh

# Prepare the SRE 2010 evaluation data.
local/make_sre_2010_test.pl /export/corpora5/SRE/SRE2010/eval/ data/
local/make_sre_2010_train.pl /export/corpora5/SRE/SRE2010/eval/ data/

# Prepare a collection of NIST SRE data prior to 2010. This is
# used to train the PLDA model and is also combined with SWB
# for UBM and i-vector extractor training data.
local/make_sre.sh data

# Prepare SWB for UBM and i-vector extractor training.
local/make_swbd2_phase2.pl /export/corpora5/LDC/LDC99S79 \
  data/swbd2_phase2_train
local/make_swbd2_phase3.pl /export/corpora5/LDC/LDC2002S06 \
  data/swbd2_phase3_train
local/make_swbd_cellular1.pl /export/corpora5/LDC/LDC2001S13 \
  data/swbd_cellular1_train
local/make_swbd_cellular2.pl /export/corpora5/LDC/LDC2004S07 \
  data/swbd_cellular2_train

utils/combine_data.sh data/train \
  data/swbd_cellular1_train data/swbd_cellular2_train \
  data/swbd2_phase2_train data/swbd2_phase3_train data/sre

cp -r data/train data/train_dnn
cp -r data/sre data/sre_dnn
cp -r data/sre10_train data/sre10_train_dnn
cp -r data/sre10_test data/sre10_test_dnn

# Extract speaker recogntion features.
steps/make_mfcc.sh --mfcc-config conf/mfcc.conf --nj 40 --cmd "$train_cmd" \
  data/train exp/make_mfcc $mfccdir
steps/make_mfcc.sh --mfcc-config conf/mfcc.conf --nj 40 --cmd "$train_cmd" \
  data/sre exp/make_mfcc $mfccdir
steps/make_mfcc.sh --mfcc-config conf/mfcc.conf --nj 40 --cmd "$train_cmd" \
  data/sre10_train exp/make_mfcc $mfccdir
steps/make_mfcc.sh --mfcc-config conf/mfcc.conf --nj 40 --cmd "$train_cmd" \
  data/sre10_test exp/make_mfcc $mfccdir

# Extract DNN features.
steps/make_mfcc.sh --mfcc-config conf/mfcc_hires.conf --nj 40 \
  --cmd "$train_cmd" data/train_dnn exp/make_mfcc $mfccdir
steps/make_mfcc.sh --mfcc-config conf/mfcc_hires.conf --nj 40 \
  --cmd "$train_cmd" data/sre_dnn exp/make_mfcc $mfccdir
steps/make_mfcc.sh --mfcc-config conf/mfcc_hires.conf --nj 40 \
  --cmd "$train_cmd" data/sre10_train_dnn exp/make_mfcc $mfccdir
steps/make_mfcc.sh --mfcc-config conf/mfcc_hires.conf --nj 40 \
  --cmd "$train_cmd" data/sre10_test_dnn exp/make_mfcc $mfccdir

for name in sre_dnn sre10_train_dnn sre10_test_dnn train_dnn sre \
  sre10_train sre10_test train; do
  utils/fix_data_dir.sh data/${name}
done

# Compute VAD decisions. These will be shared across both sets of features.
sid/compute_vad_decision.sh --nj 40 --cmd "$train_cmd" \
  data/train exp/make_vad $vaddir
sid/compute_vad_decision.sh --nj 40 --cmd "$train_cmd" \
  data/sre exp/make_vad $vaddir
sid/compute_vad_decision.sh --nj 40 --cmd "$train_cmd" \
  data/sre10_train exp/make_vad $vaddir
sid/compute_vad_decision.sh --nj 40 --cmd "$train_cmd" \
  data/sre10_test exp/make_vad $vaddir

for name in sre sre10_train sre10_test train; do
  cp data/${name}/vad.scp data/${name}_dnn/vad.scp
  cp data/${name}/utt2spk data/${name}_dnn/utt2spk
  cp data/${name}/spk2utt data/${name}_dnn/spk2utt
  utils/fix_data_dir.sh data/${name}
  utils/fix_data_dir.sh data/${name}_dnn
done

# Subset training data for faster sup-GMM initialization.
utils/subset_data_dir.sh data/train_dnn 32000 data/train_dnn_32k
utils/fix_data_dir.sh data/train_dnn_32k
utils/subset_data_dir.sh --utt-list data/train_dnn_32k/utt2spk \
  data/train data/train_32k
utils/fix_data_dir.sh data/train_32k

# Initialize a full GMM from the DNN posteriors and speaker recognition
# features. This can be used both alone, as a UBM, or to initialize the
# i-vector extractor in a DNN-based system.
sid/init_full_ubm_from_dnn.sh --cmd "$train_cmd -l mem_free=6G,ram_free=6G" \
  data/train_32k \
  data/train_dnn_32k $nnet exp/full_ubm

# Train an i-vector extractor based on just the supervised-GMM.
sid/train_ivector_extractor.sh \
  --cmd "$train_cmd -l mem_free=70G,ram_free=70G" \
  --ivector-dim 600 \
  --num-iters 5 exp/full_ubm/final.ubm data/train \
  exp/extractor_sup_gmm

# Train an i-vector extractor based on the DNN-UBM.
sid/train_ivector_extractor_dnn.sh \
  --cmd "$train_cmd -l mem_free=80G,ram_free=80G" \
  --min-post 0.015 \
  --ivector-dim 600 \
  --num-iters 5 exp/full_ubm/final.ubm $nnet \
  data/train \
  data/train_dnn \
  exp/extractor_dnn

# Extract i-vectors from the extractor with the sup-GMM UBM.
sid/extract_ivectors.sh \
  --cmd "$train_cmd -l mem_free=8G,ram_free=8G" --nj 40 \
  exp/extractor_sup_gmm data/sre10_train \
  exp/ivectors_sre10_train_sup_gmm

sid/extract_ivectors.sh \
  --cmd "$train_cmd -l mem_free=8G,ram_free=8G" --nj 40 \
  exp/extractor_sup_gmm data/sre10_test \
  exp/ivectors_sre10_test_sup_gmm

sid/extract_ivectors.sh \
  --cmd "$train_cmd -l mem_free=8G,ram_free=8G" --nj 40 \
  exp/extractor_sup_gmm data/sre \
  exp/ivectors_sre_sup_gmm

# Extract i-vectors using the extractor with the DNN-UBM.
sid/extract_ivectors_dnn.sh \
  --cmd "$train_cmd -l mem_free=10G,ram_free=10G" --nj 40 \
  exp/extractor_dnn \
  $nnet \
  data/sre10_test \
  data/sre10_test_dnn \
  exp/ivectors10_test_dnn

sid/extract_ivectors_dnn.sh
  --cmd "$train_cmd -l mem_free=10G,ram_free=10G" --nj 40 \
  exp/extractor_dnn \
  $nnet \
  data/sre10_train \
  data/sre10_train_dnn \
  exp/ivectors10_train_dnn

sid/extract_ivectors_dnn.sh
  --cmd "$train_cmd -l mem_free=10G,ram_free=10G" --nj 40 \
  exp/extractor_dnn \
  $nnet \
  data/sre \
  data/sre_dnn \
  exp/ivectors_sre_dnn

# Separate the i-vectors into male and female partitions and calculate
# i-vector means used by the scoring scripts.
local/scoring_common.sh data/sre data/sre10_train data/sre10_test \
  exp/ivectors_sre_sup_gmm exp/ivectors_sre10_train_sup_gmm \
  exp/ivectors_sre10_test_sup_gmm

local/scoring_common.sh data/sre data/sre10_train data/sre10_test \
  exp/ivectors_sre_dnn exp/ivectors_sre10_train_dnn \
  exp/ivectors_sre10_test_dnn

# The commented out scripts show how to do cosine scoring with and without
# first reducing the i-vector dimensionality with LDA. PLDA tends to work
# best, so we don't focus on the scores obtained here.
#
# local/cosine_scoring.sh data/sre10_train data/sre10_test \
#   exp/ivectors_sre10_train exp/ivectors_sre10_test $trials \
#   local/scores_gmm_2048_ind_pooled
# local/lda_scoring.sh data/sre data/sre10_train data/sre10_test \
#   exp/ivectors_sre exp/ivectors_sre10_train exp/ivectors_sre10_test \
#   $trials local/scores_gmm_2048_ind_pooled

# Create a gender independent PLDA model and do scoring with the sup-GMM system.
local/plda_scoring.sh data/sre data/sre10_train data/sre10_test \
  exp/ivectors_sre_sup_gmm exp/ivectors_sre10_train_sup_gmm \
  exp/ivectors_sre10_test_sup_gmm $trials local/scores_sup_gmm_ind_pooled
local/plda_scoring.sh --use-existing-models true data/sre data/sre10_train_female data/sre10_test_female \
  exp/ivectors_sre_sup_gmm exp/ivectors_sre10_train_sup_gmm_female \
  exp/ivectors_sre10_test_sup_gmm_female $trials_female local/scores_sup_gmm_ind_female
local/plda_scoring.sh --use-existing-models true data/sre data/sre10_train_male data/sre10_test_male \
  exp/ivectors_sre_sup_gmm exp/ivectors_sre10_train_sup_gmm_male \
  exp/ivectors_sre10_test_sup_gmm_male $trials_male local/scores_sup_gmm_ind_male

# Create gender dependent PLDA models and do scoring with the sup-GMM system.
local/plda_scoring.sh data/sre_female data/sre10_train_female data/sre10_test_female \
  exp/ivectors_sre_sup_gmm exp/ivectors_sre10_train_sup_gmm_female \
  exp/ivectors_sre10_test_sup_gmm_female $trials_female local/scores_sup_gmm_dep_female
local/plda_scoring.sh data/sre_male data/sre10_train_male data/sre10_test_male \
  exp/ivectors_sre_sup_gmm exp/ivectors_sre10_train_sup_gmm_male \
  exp/ivectors_sre10_test_sup_gmm_male $trials_male local/scores_sup_gmm_dep_male
mkdir -p local/scores_sup_gmm_dep_pooled
cat local/scores_sup_gmm_dep_male/plda_scores local/scores_sup_gmm_dep_female/plda_scores \
  > local/scores_sup_gmm_dep_pooled/plda_scores

# Create a gender independent PLDA model and do scoring with the DNN system.
local/plda_scoring.sh data/sre data/sre10_train data/sre10_test \
  exp/ivectors_sre_dnn exp/ivectors_sre10_train_dnn \
  exp/ivectors_sre10_test_dnn $trials local/scores_dnn_ind_pooled
local/plda_scoring.sh --use-existing-models true data/sre data/sre10_train_female data/sre10_test_female \
  exp/ivectors_sre_dnn exp/ivectors_sre10_train_dnn_female \
  exp/ivectors_sre10_test_dnn_female $trials_female local/scores_dnn_ind_female
local/plda_scoring.sh --use-existing-models true data/sre data/sre10_train_male data/sre10_test_male \
  exp/ivectors_sre_dnn exp/ivectors_sre10_train_dnn_male \
  exp/ivectors_sre10_test_dnn_male $trials_male local/scores_dnn_ind_male

# Create gender dependent PLDA models and do scoring with the DNN system.
local/plda_scoring.sh data/sre_female data/sre10_train_female data/sre10_test_female \
  exp/ivectors_sre_dnn exp/ivectors_sre10_train_dnn_female \
  exp/ivectors_sre10_test_dnn_female $trials_female local/scores_dnn_dep_female
local/plda_scoring.sh data/sre_male data/sre10_train_male data/sre10_test_male \
  exp/ivectors_sre_dnn exp/ivectors_sre10_train_dnn_male \
  exp/ivectors_sre10_test_dnn_male $trials_male local/scores_dnn_dep_male
mkdir -p local/scores_dnn_dep_pooled
cat local/scores_dnn_dep_male/plda_scores local/scores_dnn_dep_female/plda_scores \
  > local/scores_dnn_dep_pooled/plda_scores

# Sup-GMM PLDA EER
# ind pooled: 1.72
# ind female: 1.81
# ind male:   1.56
# dep female: 1.89
# dep male:   1.39
# dep pooled: 1.65
echo "Sup-GMM-$num_components EER"
for x in ind dep; do
  for y in female male pooled; do
    eer=`compute-eer <(python local/prepare_for_eer.py $trials local/scores_sup_gmm_${x}_${y}/plda_scores) 2> /dev/null`
    echo "${x} ${y}: $eer"
  done
done

# DNN PLDA EER
# ind pooled: 1.05
# ind female: 1.33
# ind male:   0.75
# dep female: 1.41
# dep male:   0.64
# dep pooled: 1.02
echo "DNN-$num_components EER"
for x in ind dep; do
  for y in female male pooled; do
    eer=`compute-eer <(python local/prepare_for_eer.py $trials local/scores_dnn_${x}_${y}/plda_scores) 2> /dev/null`
    echo "${x} ${y}: $eer"
  done
done

# In comparison, here is the EER for an unsupervised GMM-based system
# with 5297 components (the same as the number of senones in the DNN):
# GMM-5297 PLDA EER
# ind pooled: 2.25
# ind female: 2.33
# ind male:   2.08
# dep female: 2.25
# dep male:   1.50
# dep pooled: 1.91
