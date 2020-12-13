#!/bin/bash

. ./cmd.sh
. ./path.sh

# Train systems,
nj=30 # number of parallel jobs,
stage=0
. utils/parse_options.sh

base_mic=$(echo $mic | sed 's/[0-9]//g') # sdm, ihm or mdm
nmics=$(echo $mic | sed 's/[a-z]//g') # e.g. 8 for mdm8.

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
  echo ============================================================================
  echo "              Prepare ICSI data"
  echo ============================================================================
  #prepare annotations, note: dict is assumed to exist when this is called
  local/ICSI/icsi_run_prepare_shared.sh
  local/ICSI/icsi_ihm_data_prep.sh $ICSI_DIR
  local/ICSI/icsi_ihm_scoring_data_prep.sh $ICSI_DIR dev
  local/ICSI/icsi_ihm_scoring_data_prep.sh $ICSI_DIR eval
fi

if [ $stage -le 3 ]; then
  echo ============================================================================
  echo "              Prepare AMI data"
  echo ============================================================================
  local/AMI/ami_text_prep.sh data/local/download
  local/AMI/ami_ihm_data_prep.sh $AMI_DIR
  local/AMI/ami_ihm_scoring_data_prep.sh $AMI_DIR dev
  local/AMI/ami_ihm_scoring_data_prep.sh $AMI_DIR eval
fi

if [ $stage -le 4 ]; then
  for dset in train dev eval; do
    utils/data/modify_speaker_info.sh --seconds-per-spk-max 30 \
      data/AMI/${dset}_orig data/AMI/$dset
    utils/data/modify_speaker_info.sh --seconds-per-spk-max 30 \
      data/ICSI/${dset}_orig data/ICSI/$dset
  done
fi

if [ $stage -le 5 ]; then
  utils/data/get_reco2dur.sh data/AMI/train
  utils/data/get_reco2dur.sh data/ICSI/train

  utils/data/get_utt2dur.sh data/AMI/train
  utils/data/get_utt2dur.sh data/ICSI/train

  for dataset in AMI ICSI; do
    for split in train dev eval; do
      cat data/$dataset/$split/text | awk '{printf $1""FS;for(i=2; i<=NF; ++i) printf "%s",tolower($i)""FS; print""}'  > data/$dataset/$split/texttmp
      mv data/$dataset/$split/text data/$dataset/$split/textupper
      mv data/$dataset/$split/texttmp data/$dataset/$split/text
    done
  done
fi

if [ $stage -le 6 ]; then
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

  utils/data/combine_data.sh data/train_safet data/safe_t_r20 data/safe_t_r11
fi

if [ $stage -le 7 ] ; then
  echo ============================================================================
  echo "              Obtain LM from SAFE-T train text"
  echo ============================================================================
  local/safet_train_lms_srilm.sh \
    --train_text data/train_safet/text --dev_text data/safe_t_dev1/text  \
    data/ data/local/srilm
  utils/format_lm.sh  data/lang_nosp/ data/local/srilm/lm.gz\
    data/local/lexicon.txt  data/lang_nosp_test
fi

if [ $stage -le 8 ] ; then
  echo ============================================================================
  echo "              Extract features"
  echo ============================================================================
  utils/data/combine_data.sh data/train_icsiami data/ICSI/train data/AMI/train
  for dset in train_icsiami train_safet; do
    steps/make_mfcc.sh --nj 75 --cmd "$train_cmd" data/$dset
    steps/compute_cmvn_stats.sh data/$dset
    utils/fix_data_dir.sh data/$dset
  done
  utils/data/combine_data.sh data/train_all data/train_safet data/train_icsiami
fi

suffix=all
# monophone training
if [ $stage -le 9 ]; then
  echo ============================================================================
  echo "        GMM-HMM training : mono, delta, LDA + MLLT + SAT"
  echo ============================================================================
  utils/subset_data_dir.sh data/train_${suffix} 15000 data/train_15k
  steps/train_mono.sh --nj $nj --cmd "$train_cmd" \
    data/train_15k data/lang_nosp_test exp/mono_train_${suffix}
  steps/align_si.sh --nj $nj --cmd "$train_cmd" \
    data/train_${suffix} data/lang_nosp_test exp/mono_train_${suffix} exp/mono_train_${suffix}_ali
fi

# context-dep. training with delta features.
if [ $stage -le 10 ]; then
  steps/train_deltas.sh --cmd "$train_cmd" \
    5000 80000 data/train_$suffix data/lang_nosp_test exp/mono_train_${suffix}_ali exp/tri1_train_${suffix}
  steps/align_si.sh --nj $nj --cmd "$train_cmd" \
    data/train_${suffix} data/lang_nosp_test exp/tri1_train_${suffix} exp/tri1_train_${suffix}_ali
fi

if [ $stage -le 11 ]; then
  steps/train_lda_mllt.sh --cmd "$train_cmd" \
    --splice-opts "--left-context=3 --right-context=3" \
    5000 80000 data/train_$suffix data/lang_nosp_test exp/tri1_train_${suffix}_ali exp/tri2_train_${suffix}
  steps/align_fmllr.sh --nj $nj --cmd "$train_cmd" \
    data/train_${suffix} data/lang_nosp_test exp/tri2_train_${suffix} exp/tri2_train_${suffix}_ali
fi

if [ $stage -le 12 ]; then
  steps/train_sat.sh --cmd "$train_cmd" \
    5000 80000 data/train_$suffix data/lang_nosp_test exp/tri2_train_${suffix}_ali exp/tri3_train_${suffix}
  steps/align_fmllr.sh --nj $nj --cmd "$train_cmd" \
    data/train_${suffix} data/lang_nosp_test exp/tri3_train_${suffix} exp/tri3_train_${suffix}_ali
fi

if [ $stage -le 13 ]; then
  echo ============================================================================
  echo "              augmentation, i-vector extraction, and chain model training"
  echo ============================================================================
  local/chain/run_cnn_tdnn_shared.sh
fi

echo ===========================================================================
echo "Finished Successfully"
echo ===========================================================================
exit 0
