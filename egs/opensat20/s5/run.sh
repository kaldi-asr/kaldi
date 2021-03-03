#!/bin/bash

. ./cmd.sh
. ./path.sh

# Train systems,
nj=30 # number of parallel jobs,
stage=0
. utils/parse_options.sh

set -euo pipefail
opensat_corpora=/export/corpora5/opensat_corpora
SAFE_T_AUDIO_R20=/export/corpora5/opensat_corpora/LDC2020E10
SAFE_T_TEXTS_R20=/export/corpora5/opensat_corpora/LDC2020E09
SAFE_T_AUDIO_R11=/export/corpora5/opensat_corpora/LDC2019E37
SAFE_T_TEXTS_R11=/export/corpora5/opensat_corpora/LDC2019E36
SAFE_T_AUDIO_DEV1=/export/corpora5/opensat_corpora/LDC2019E53
SAFE_T_TEXTS_DEV1=/export/corpora5/opensat_corpora/LDC2019E53
SAFE_T_AUDIO_EVAL1=/export/corpora5/opensat_corpora/LDC2020E07
ICSI_DIR=/export/corpora5/LDC/LDC2004S02/meeting_speech/speech
AMI_DIR=/export/corpora5/amicorpus/

if [ $stage -le 0 ]; then
  echo ============================================================================
  echo "              Prepare SAFE-T data"
  echo ============================================================================
  local/safet_data_prep.sh ${SAFE_T_AUDIO_R11} ${SAFE_T_TEXTS_R11} data/safe_t_r11
  local/safet_data_prep.sh ${SAFE_T_AUDIO_R20} ${SAFE_T_TEXTS_R20} data/safe_t_r20
  local/safet_data_prep.sh ${SAFE_T_AUDIO_DEV1} ${SAFE_T_TEXTS_DEV1} data/safe_t_dev1
  local/safet_data_prep.sh ${SAFE_T_AUDIO_EVAL1} data/safe_t_eval1
fi

if [ $stage -le 1 ]; then
  local/safet_get_cmu_dict.sh
  utils/prepare_lang.sh data/local/dict_nosp '<UNK>' data/local/lang_nosp data/lang_nosp
  utils/validate_lang.pl data/lang_nosp
fi

if [ $stage -le 2 ]; then
  mkdir -p exp/cleanup_stage_1
  (
    local/safet_cleanup_transcripts.py data/local/lexicon.txt data/safe_t_r11/transcripts data/safe_t_r11/transcripts.clean
    local/safet_cleanup_transcripts.py data/local/lexicon.txt data/safe_t_r20/transcripts data/safe_t_r20/transcripts.clean
  ) | sort > exp/cleanup_stage_1/oovs

  local/safet_cleanup_transcripts.py --no-unk-replace  data/local/lexicon.txt \
    data/safe_t_dev1/transcripts data/safe_t_dev1/transcripts.clean > exp/cleanup_stage_1/oovs.dev1

  local/safet_build_data_dir.sh data/safe_t_r11/ data/safe_t_r11/transcripts.clean
  local/safet_build_data_dir.sh data/safe_t_r20/ data/safe_t_r20/transcripts.clean
  local/safet_build_data_dir.sh data/safe_t_dev1/ data/safe_t_dev1/transcripts

  utils/data/combine_data.sh data/train data/safe_t_r20 data/safe_t_r11
fi

if [ $stage -le 3 ] ; then
  echo ============================================================================
  echo "              Obtain LM from SAFE-T train text"
  echo ============================================================================
  local/safet_train_lms_srilm.sh \
    --train_text data/train/text --dev_text data/safe_t_dev1/text  \
    data/ data/local/srilm
  utils/format_lm.sh  data/lang_nosp/ data/local/srilm/lm.gz\
    data/local/lexicon.txt  data/lang_nosp_test
fi

# Feature extraction,
if [ $stage -le 4 ]; then
  echo ============================================================================
  echo "              Extract features"
  echo ============================================================================
  steps/make_mfcc.sh --nj 75 --cmd "$train_cmd" data/train
  steps/compute_cmvn_stats.sh data/train
  utils/fix_data_dir.sh data/train
fi

# monophone training
if [ $stage -le 5 ]; then
  echo ============================================================================
  echo "        GMM-HMM training : mono, delta, LDA + MLLT + SAT"
  echo ============================================================================
  utils/subset_data_dir.sh data/train 15000 data/train_15k
  steps/train_mono.sh --nj $nj --cmd "$train_cmd" \
    data/train data/lang_nosp_test exp/mono_train
  steps/align_si.sh --nj $nj --cmd "$train_cmd" \
    data/train data/lang_nosp_test exp/mono_train exp/mono_train_ali
fi

# context-dep. training with delta features.
if [ $stage -le 6 ]; then
  steps/train_deltas.sh --cmd "$train_cmd" \
    5000 80000 data/train data/lang_nosp_test exp/mono_train_ali exp/tri1_train
  steps/align_si.sh --nj $nj --cmd "$train_cmd" \
    data/train data/lang_nosp_test exp/tri1_train exp/tri1_train_ali
fi

if [ $stage -le 7 ]; then
  steps/train_lda_mllt.sh --cmd "$train_cmd" \
    --splice-opts "--left-context=3 --right-context=3" \
    5000 80000 data/train data/lang_nosp_test exp/tri1_train_ali exp/tri2_train
  steps/align_fmllr.sh --nj $nj --cmd "$train_cmd" \
    data/train data/lang_nosp_test exp/tri2_train exp/tri2_train_ali
fi

if [ $stage -le 8 ]; then
  steps/train_sat.sh --cmd "$train_cmd" \
    5000 80000 data/train data/lang_nosp_test exp/tri2_train_ali exp/tri3_train
  steps/align_fmllr.sh --nj $nj --cmd "$train_cmd" \
    data/train data/lang_nosp_test exp/tri3_train exp/tri3_train_ali
fi

if [ $stage -le 9 ]; then
  echo ============================================================================
  echo "              augmentation, i-vector extraction, and chain model training"
  echo ============================================================================
  local/chain/run_cnn_tdnn.sh
fi

echo ===========================================================================
echo "Finished Successfully"
echo ===========================================================================
exit 0
