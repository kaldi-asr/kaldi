#!/bin/bash
# Copyright      2017   Speech and Multimodal Intelligent Information Processing Lab, SYSU (Author: Danwei Cai)
#                2017   Speech and Multimodal Intelligent Information Processing Lab, SYSU (Author: Ming Li)
#           2015-2016   David Snyder
#                2015   Johns Hopkins University (Author: Daniel Garcia-Romero)
#                2015   Johns Hopkins University (Author: Daniel Povey)
# Apache 2.0.
#
# See README.txt for more info on data required.
# Results (EERs) are inline in comments below.
#

. cmd.sh
. path.sh
set -e
mfccdir=`pwd`/mfcc
vaddir=`pwd`/mfcc
trials_female=data/sre10_test_tandem_female/trials
trials_male=data/sre10_test_tandem_male/trials
trials=data/sre10_test_tandem/trials
num_components=2048 # Larger than this doesn't make much of a difference.
nnet=exp/nnet2_online/nnet_ms_a/final.mdl

# Train a DNN on about 1800 hours of the english portion of Fisher.
local/dnn/train_dnn.sh || exit 1;

# Prepare the SRE 2010 evaluation data.
local/make_sre_2010_test.pl /export/corpora5/SRE/SRE2010/eval/ data/ || exit 1;
local/make_sre_2010_train.pl /export/corpora5/SRE/SRE2010/eval/ data/ || exit 1;

# Prepare a collection of NIST SRE data prior to 2010. This is
# used to train the PLDA model and is also combined with SWB
# for UBM and i-vector extractor training data.
local/make_sre.sh data || exit 1;

# Prepare SWB for UBM and i-vector extractor training.
local/make_swbd2_phase2.pl /export/corpora5/LDC/LDC99S79 \
  data/swbd2_phase2_train || exit 1;
local/make_swbd2_phase3.pl /export/corpora5/LDC/LDC2002S06 \
  data/swbd2_phase3_train || exit 1;
local/make_swbd_cellular1.pl /export/corpora5/LDC/LDC2001S13 \
  data/swbd_cellular1_train || exit 1;
local/make_swbd_cellular2.pl /export/corpora5/LDC/LDC2004S07 \
  data/swbd_cellular2_train || exit 1;

for name in swbd2_phase2_train swbd2_phase3_train swbd_cellular1_train \
  swbd_cellular2_train sre sre10_train sre10_test; do
  # Extract speaker recogntion features.
  steps/make_mfcc.sh --mfcc-config conf/mfcc.conf --nj 32 --cmd "$train_cmd" \
    data/${name} exp/make_mfcc $mfccdir || exit 1;
  utils/fix_data_dir.sh data/${name} || exit 1;

  # Extract DNN features.
  cp -r data/$name data/${name}_hires || exit 1;
  steps/make_mfcc.sh --mfcc-config conf/mfcc_hires.conf --nj 32 --cmd "$train_cmd" \
    data/${name}_hires exp/make_mfcc_hires $mfcchiresdir || exit 1;
  utils/fix_data_dir.sh data/${name}_hires || exit 1;

  # Compute VAD decisions. These will be shared across both sets of features.
  sid/compute_vad_decision.sh --nj 32 --cmd "$train_cmd" \
    data/${name} exp/make_vad $vaddir || exit 1;

  cp data/${name}/vad.scp data/${name}_hires/vad.scp || exit 1;
  cp data/${name}/utt2spk data/${name}_hires/utt2spk || exit 1;
  cp data/${name}/spk2utt data/${name}_hires/spk2utt || exit 1;
  utils/fix_data_dir.sh data/${name} || exit 1;
  utils/fix_data_dir.sh data/${name}_hires || exit 1;
done

# Prepare a data collection to extract Phone Posterior Probability for PCA training
utils/combine_data.sh --extra_files vad.scp data/train_pca \
  data/swbd_cellular1_train_hires data/swbd_cellular2_train_hires \
  data/swbd2_phase2_train_hires data/swbd2_phase3_train_hires \
  data/sre_hires || exit 1;
utils/shuffle_list.pl data/train_pca/feats.scp | head -n 1000 | sort \
  > data/train_pca/feats2.scp || exit 1;
mv data/train_pca/feats2.scp data/train_pca/feats.scp
utils/fix_data_dir.sh data/train_pca || exit 1;

# Extract PPP (Phone Posterior Probability) for PCA training
local/make_ppp.sh --nj 10 --use_gpu yes --chunk_size 1024 $nnet \
  data/train_pca exp/make_ppp_for_pca || exit 1;

#Train PCA transform matrix
local/train_pca.sh data/train_pca exp/train_pca || exit 1;

# Extract tandem features
# sre10_train
for name in sre10_train sre10_test swbd_cellular1_train swbd_cellular2_train \
  swbd2_phase2_train swbd2_phase3_train sre; do
  cp -r data/${name} data/${name}_tandem || exit 1;
  local/make_tandem.sh --nj 15 --use_gpu yes --chunk_size 2048 $nnet \
    exp/train_pca/final.mat data/${name} data/${name}_hires \
    exp/make_tandem data/${name}_tandem || exit 1;
done

utils/combine_data.sh data/train_tandem data/swbd_cellular1_train_tandem \
  data/swbd_cellular2_train_tandem data/swbd2_phase2_train_tandem \
  data/swbd2_phase3_train_tandem data/sre_tandem || exit 1;

# Reduce the amount of training data for the UBM.
utils/subset_data_dir.sh data/train_tandem 16000 data/train_tandem_16k || exit 1;
utils/subset_data_dir.sh data/train_tandem 32000 data/train_tandem_32k || exit 1;

# Train UBM and i-vector extractor.
sid/train_diag_ubm_v3.sh --cmd "$train_cmd" \
  --nj 32 --num-threads 32 \
  data/train_tandem_16k $num_components \
  exp/diag_ubm_$num_components || exit 1;

sid/train_full_ubm_v3.sh --nj 32 --remove-low-count-gaussians false \
  --cmd "$train_cmd" data/train_tandem_32k \
  exp/diag_ubm_$num_components exp/full_ubm_$num_components || exit 1;

sid/train_ivector_extractor_v3.sh --cmd "$train_cmd"\
  --ivector-dim 600 --nj 1 --num_threads 8 --num_processes 4 \
  --num-iters 5 exp/full_ubm_$num_components/final.ubm data/train_tandem \
  exp/extractor || exit 1;

# Extract i-vectors.
sid/extract_ivectors_v3.sh --cmd "$train_cmd" --nj 32 \
  exp/extractor data/sre10_train_tandem exp/ivectors_sre10_train || exit 1;

sid/extract_ivectors_v3.sh --cmd "$train_cmd" --nj 32 \
  exp/extractor data/sre10_test_tandem exp/ivectors_sre10_test || exit 1;

sid/extract_ivectors_v3.sh --cmd "$train_cmd" --nj 32 \
  exp/extractor data/sre_tandem exp/ivectors_sre || exit 1;

# Separate the i-vectors into male and female partitions and calculate
# i-vector means used by the scoring scripts.
local/scoring_common.sh data/sre_tandem data/sre10_train_tandem data/sre10_test_tandem \
  exp/ivectors_sre exp/ivectors_sre10_train exp/ivectors_sre10_test

# Create a gender independent PLDA model and do scoring with the DNN system.
local/plda_scoring.sh data/sre_tandem data/sre10_train_tandem data/sre10_test_tandem \
  exp/ivectors_sre exp/ivectors_sre10_train exp/ivectors_sre10_test \
  $trials local/scores_gmm_2048_ind_pooled
local/plda_scoring.sh --use-existing-models true \
  data/sre_tandem data/sre10_train_tandem_female data/sre10_test_tandem_female \
  exp/ivectors_sre exp/ivectors_sre10_train_female exp/ivectors_sre10_test_female \
  $trials_female local/scores_gmm_2048_ind_female
local/plda_scoring.sh --use-existing-models true \
  data/sre_tandem data/sre10_train_tandem_male data/sre10_test_tandem_male \
  exp/ivectors_sre exp/ivectors_sre10_train_male exp/ivectors_sre10_test_male \
  $trials_male local/scores_gmm_2048_ind_male

# Create gender dependent PLDA models and do scoring.
local/plda_scoring.sh data/sre_tandem_female data/sre10_train_tandem_female \
  data/sre10_test_tandem_female exp/ivectors_sre exp/ivectors_sre10_train_female \
  exp/ivectors_sre10_test_female $trials_female local/scores_gmm_2048_dep_female
local/plda_scoring.sh data/sre_tandem_male data/sre10_train_tandem_male \
  data/sre10_test_tandem_male exp/ivectors_sre exp/ivectors_sre10_train_male \
  exp/ivectors_sre10_test_male $trials_male local/scores_gmm_2048_dep_male

mkdir -p local/scores_gmm_2048_dep_pooled
cat local/scores_gmm_2048_dep_male/plda_scores local/scores_gmm_2048_dep_female/plda_scores \
  > local/scores_gmm_2048_dep_pooled/plda_scores

# GMM-2048 tandem(MFCC+DNN PPP/PCA) / DNN-UBM EER
#             v3       v2
#ind female:  0.9179   1.16
#ind male:    0.7215   0.78
#ind pooled:  0.823    1.01
#dep female:  1.053    1.27
#dep male:    0.6349   0.61
#dep pooled:  0.8509   0.96

echo "GMM-$num_components EER"
for x in ind dep; do
  for y in female male pooled; do
    eer=`compute-eer <(python local/prepare_for_eer.py $trials local/scores_gmm_${num_components}_${x}_${y}/plda_scores) 2> /dev/null`
    echo "${x} ${y}: $eer"
  done
done

