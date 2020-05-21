#!/usr/bin/env bash
#
# This script decodes raw utterances through the entire pipeline:
# Feature extraction -> SAD -> Diarization -> ASR
#
# Copyright  2017  Johns Hopkins University (Author: Shinji Watanabe and Yenda Trmal)
#            2019  Desh Raj, David Snyder, Ashish Arora, Zhaoheng Ni
# Apache 2.0

# Begin configuration section.
nj=8
stage=0
score_sad=true
diarizer_stage=4
decode_diarize_stage=0
decode_oracle_stage=1
score_stage=0

# If the following is set to true, we use the oracle speaker and segment
# information instead of performing SAD and diarization.
use_oracle_segments=

test_sets="dev eval"

. ./utils/parse_options.sh

. ./cmd.sh
. ./path.sh

$use_oracle_segments && [ $stage -le 6 ] && stage=6

#######################################################################
# Perform SAD on the dev/eval data using py-webrtcvad package
#######################################################################

if [ $stage -le 1 ]; then
  for datadir in ${test_sets}; do
    echo "Applying WEBRTC-VAD on ${datadir}"
    test_set=data/${datadir}
    if [ ! -f ${test_set}/wav.scp ]; then
      echo "$0: Not performing SAD on ${test_set}"
      exit 0
    fi

    # Perform segmentation
    local/segmentation/apply_webrtcvad.py --mode 0 $test_set | sort > $test_set/segments
    
    # Create dummy utt2spk file from obtained segments
    awk '{print $1, $2}' ${test_set}/segments > ${test_set}/utt2spk
    utils/utt2spk_to_spk2utt.pl ${test_set}/utt2spk > ${test_set}/spk2utt
    
    # Generate RTTM file from segmentation performed by SAD. This can
    # be used to evaluate the performance of the SAD as an intermediate
    # step.
    steps/segmentation/convert_utt2spk_and_segments_to_rttm.py \
      ${test_set}/utt2spk ${test_set}/segments ${test_set}/rttm

    if [ $score_sad == "true" ]; then
      echo "Scoring $datadir.."
      # We first generate the reference RTTM from the backed up utt2spk and segments
      # files.
      ref_rttm=${test_set}/ref_rttm
      steps/segmentation/convert_utt2spk_and_segments_to_rttm.py ${test_set}/utt2spk.bak \
        ${test_set}/segments.bak ${test_set}/ref_rttm

      md-eval.pl -r $ref_rttm -s ${test_set}/rttm |\
        awk 'or(/MISSED SPEECH/,/FALARM SPEECH/)'
    fi
    
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
    diar_nj=$(wc -l < "data/$datadir/wav.scp")

    local/diarize.sh --nj $diar_nj --cmd "$train_cmd" --stage $diarizer_stage \
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
    local/decode_diarized.sh --nj $nj --cmd "$decode_cmd" --stage $decode_diarize_stage \
      exp/${datadir}_${nnet_type}_seg_diarization data/$datadir data/lang \
      exp/chain_${train_set}_cleaned_rvb exp/nnet3_${train_set}_cleaned_rvb \
      data/${datadir}_diarized || exit 1
  done
fi

#######################################################################
# Score decoded dev/eval sets
#######################################################################
# if [ $stage -le 5 ]; then
#   # TODO
# fi

$use_oracle_segments || exit 0

######################################################################
# Here we decode using oracle speaker and segment information
######################################################################
if [ $stage -le 6 ]; then
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

if [ $stage -le 7 ]; then
  local/decode_oracle.sh --stage $decode_oracle_stage \
    --lang-dir data/lang_nosp_test_tgsmall \
    --lm-suffix "_tgsmall" \
    --test_sets "$test_sets"
fi

exit 0;
