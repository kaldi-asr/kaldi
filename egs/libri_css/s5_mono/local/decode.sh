#!/usr/bin/env bash
#
# This script decodes raw utterances through the entire pipeline:
# VAD -> Feature extraction -> Diarization -> ASR
#
# Copyright  2017  Johns Hopkins University (Author: Shinji Watanabe and Yenda Trmal)
#            2019  Desh Raj, David Snyder, Ashish Arora, Zhaoheng Ni
# Apache 2.0

# Begin configuration section.
nj=8
stage=0

diarizer_stage=1
decode_diarize_stage=0
decode_oracle_stage=0
score_stage=0
nnet3_affix=_cleaned # affix for the chain directory name
affix=1d_ft   # affix for the TDNN directory name

# If the following is set to true, we use the oracle speaker and segment
# information instead of performing SAD and diarization.
use_oracle_segments=
sad_type=webrtc # Set this to webrtc or tdnn
rnnlm_rescore=true

# RNNLM rescore options
ngram_order=4 # approximate the lattice-rescoring by limiting the max-ngram-order
              # if it's set, it merges histories in the lattice if they share
              # the same ngram history and this prevents the lattice from 
              # exploding exponentially
pruned_rescore=true
rnnlm_dir=exp/rnnlm_lstm_1a

test_sets="dev eval"

. ./utils/parse_options.sh

. ./cmd.sh
. ./path.sh

# Get dev and eval set names from the test_sets
dev_set=$( echo $test_sets | cut -d " " -f1 )
eval_set=$( echo $test_sets | cut -d " " -f2 )

$use_oracle_segments && [ $stage -le 8 ] && stage=8

#######################################################################
# Perform SAD on the dev/eval data using py-webrtcvad package
#######################################################################
if [ $stage -le 1 ]; then
  for datadir in ${test_sets}; do
    test_set=data/${datadir}
    if [ $sad_type == "webrtc" ]; then
      echo "Applying WebRTC-VAD on ${datadir}"
      local/segmentation/apply_webrtcvad.py --mode 0 $test_set | sort > $test_set/segments
    else
      echo "Applying TDNN-Stats-SAD on ${datadir}"
      if [ ! -f ${test_set}/wav.scp ]; then
        echo "$0: Not performing SAD on ${test_set}, since wav.scp does not exist. Exiting!"
        exit 0
      fi

      sad_nj=$(wc -l < "$test_set/wav.scp")
      nj=$((decode_nj>sad_nj ? sad_nj : decode_nj))
      # Perform segmentation. We use the pretrained CHiME-6 SAD available at:
      # http://kaldi-asr.org/models/12/0012_sad_v1.tar.gz
      # Download and extract using tar -xvzf
      if [ ! -d exp/segmentation_1a/tdnn_stats_sad_1a ]; then
        wget http://kaldi-asr.org/models/12/0012_sad_v1.tar.gz || exit
        tar -xvzf 0012_sad_v1.tar.gz
        cp -r 0012_sad_v1/conf/* conf/
        cp -r 0012_sad_v1/exp/segmentation_1a exp/
      fi
      local/detect_speech_activity.sh --cmd "$decode_cmd" --nj $sad_nj $test_set \
        exp/segmentation_1a/tdnn_stats_sad_1a
    fi

    # Create dummy utt2spk file from obtained segments
    awk '{print $1, $2}' ${test_set}/segments > ${test_set}/utt2spk
    utils/utt2spk_to_spk2utt.pl ${test_set}/utt2spk > ${test_set}/spk2utt
    
    steps/segmentation/convert_utt2spk_and_segments_to_rttm.py \
      ${test_set}/utt2spk ${test_set}/segments ${test_set}/rttm

    echo "Scoring $datadir.."
    # We first generate the reference RTTM from the backed up utt2spk and segments
    # files.
    ref_rttm=${test_set}/ref_rttm
    steps/segmentation/convert_utt2spk_and_segments_to_rttm.py ${test_set}/utt2spk.bak \
      ${test_set}/segments.bak ${test_set}/ref_rttm

    md-eval.pl -r $ref_rttm -s ${test_set}/rttm |\
      awk '/(MISSED|FALARM) SPEECH/'
    
  done
fi

#######################################################################
# Feature extraction for the dev and eval data
#######################################################################
if [ $stage -le 2 ]; then
  # mfccdir should be some place with a largish disk where you
  # want to store MFCC features.
  mfccdir=mfcc
  for x in ${test_sets}; do
    steps/make_mfcc.sh --nj $nj --cmd "$train_cmd" \
      --mfcc-config conf/mfcc_hires.conf \
      data/$x exp/make_mfcc/$x $mfccdir
  done
fi

#######################################################################
# Perform diarization on the dev/eval data
#######################################################################
if [ $stage -le 3 ]; then
  for datadir in ${test_sets}; do
    ref_rttm=data/${datadir}/ref_rttm
    steps/segmentation/convert_utt2spk_and_segments_to_rttm.py data/${datadir}/utt2spk.bak \
      data/${datadir}/segments.bak $ref_rttm
    diar_nj=$(wc -l < "data/$datadir/wav.scp") # This is important especially for VB-HMM

    [ ! -d exp/xvector_nnet_1a ] && ./local/download_diarizer.sh

    local/diarize_spectral.sh --nj $diar_nj --cmd "$train_cmd" --stage $diarizer_stage \
      --ref-rttm $ref_rttm \
      exp/xvector_nnet_1a \
      data/${datadir} \
      exp/${datadir}_diarization
  done
fi

#######################################################################
# Decode diarized output using trained chain model
#######################################################################
if [ $stage -le 4 ]; then
  for datadir in ${test_sets}; do
    asr_nj=$(wc -l < "data/$datadir/wav.scp")
    local/decode_diarized.sh --nj $asr_nj --cmd "$decode_cmd" --stage $decode_diarize_stage \
      --lm-suffix "_tgsmall" \
      data/${datadir}/rttm_tsvad data/$datadir data/lang_test_tgsmall \
      exp/chain${nnet3_affix}/tdnn_${affix} exp/nnet3${nnet3_affix} \
      data/${datadir}_diarized || exit 1
  done
fi

#######################################################################
# Score decoded dev/eval sets
#######################################################################
if [ $stage -le 5 ]; then
  # please specify both dev and eval set directories so that the search parameters
  # (insertion penalty and language model weight) will be tuned using the dev set
  local/score_reco_diarized.sh --cmd "$train_cmd" --stage $score_stage \
      --dev_decodedir exp/chain${nnet3_affix}/tdnn_${affix}_sp/decode_${dev_set}_diarized_2stage \
      --dev_datadir ${dev_set}_diarized_hires \
      --eval_decodedir exp/chain${nnet3_affix}/tdnn_${affix}_sp/decode_${eval_set}_diarized_2stage \
      --eval_datadir ${eval_set}_diarized_hires
fi

############################################################################
# RNNLM rescoring
############################################################################
if $rnnlm_rescore; then
  if [ $stage -le 6 ]; then
    echo "$0: Perform RNNLM lattice-rescoring"
    pruned=
    ac_model_dir=exp/chain${nnet3_affix}/tdnn_${affix}
    if $pruned_rescore; then
      pruned=_pruned
    fi
    for decode_set in $test_sets; do
      decode_dir=${ac_model_dir}/decode_${decode_set}_diarized_2stage
      # Lattice rescoring
      rnnlm/lmrescore${pruned}.sh \
          --cmd "$decode_cmd --mem 8G" \
          --weight 0.45 --max-ngram-order $ngram_order \
          data/lang_test_tgsmall $rnnlm_dir \
          data/${decode_set}_diarized_hires ${decode_dir} \
          ${ac_model_dir}/decode_${decode_set}_diarized_2stage_rescore
    done
  fi

  if [ $stage -le 7 ]; then
    echo "$0: WERs after rescoring with $rnnlm_dir"
    local/score_reco_diarized.sh --cmd "$train_cmd" --stage $score_stage \
        --dev_decodedir exp/chain${nnet3_affix}/tdnn_${affix}/decode_${dev_set}_diarized_2stage_rescore \
        --dev_datadir ${dev_set}_diarized_hires \
        --eval_decodedir exp/chain${nnet3_affix}/tdnn_${affix}/decode_${eval_set}_diarized_2stage_rescore \
        --eval_datadir ${eval_set}_diarized_hires
  fi
fi

$use_oracle_segments || exit 0

######################################################################
# Here we decode using oracle speaker and segment information
######################################################################
if [ $stage -le 8 ]; then
  # mfccdir should be some place with a largish disk where you
  # want to store MFCC features.
  mfccdir=mfcc
  for x in ${test_sets}; do
    datadir=data/${x}_oracle
    mkdir -p $datadir
    
    cp data/$x/wav.scp $datadir/
    cp data/$x/segments.bak $datadir/segments
    cp data/$x/utt2spk.bak $datadir/utt2spk
    cp data/$x/text.bak $datadir/text
    utils/utt2spk_to_spk2utt.pl $datadir/utt2spk > $datadir/spk2utt

    steps/make_mfcc.sh --nj $nj --cmd "$train_cmd" \
      --mfcc-config conf/mfcc_hires.conf \
      $datadir exp/make_mfcc/$x $mfccdir
  done
fi

if [ $stage -le 9 ]; then
  local/decode_oracle.sh --stage $decode_oracle_stage \
    --affix $affix \
    --lang-dir data/lang_test_tgsmall \
    --lm-suffix "_tgsmall" \
    --rnnlm-rescore $rnnlm_rescore \
    --test_sets "$test_sets"
fi

exit 0;
