#!/usr/bin/env bash
#
# LibriCSS pipeline containing speech separation. We don't provide
# any training stages for diarization or ASR here, since they are 
# the same as those in s5_mono. As such, this run script is
# actually a decoding script. Before running this script, you
# need to have run your separation module (or use the separated
# audio streams we have provided), and the output streams must
# be named in the following naming convention:
# overlap_ratio_10.0_sil0.1_1.0_session7_actual10.1_channel_1.wav
# Here, "channel" denotes the stream number, for example, if your
# method separates the audio into 3 streams, they should be named
# channel_0, channel_1, and channel_2. The wav files can be organized
# in any hierarchy within the directory, but this naming
# convention must be followed.
# 
# Copyright  2020  Johns Hopkins University (Author: Desh Raj)
# Apache 2.0

# Begin configuration section.
nj=50
decode_nj=20
stage=0

nnet3_affix=_cleaned
affix=1d_ft
data_affix= # This can be used to distinguish between different data sources
sad_type=tdnn # Set this to webrtc or tdnn

# Different stages
sad_stage=0
diarizer_stage=1
decode_diarize_stage=0
score_stage=0
rnnlm_rescore=true

# RNNLM rescore options
ngram_order=4 # approximate the lattice-rescoring by limiting the max-ngram-order
              # if it's set, it merges histories in the lattice if they share
              # the same ngram history and this prevents the lattice from 
              # exploding exponentially
pruned_rescore=true
rnnlm_dir=exp/rnnlm_lstm_1a

set -e # exit on error

# End configuration section
. ./utils/parse_options.sh

. ./cmd.sh
. ./path.sh

test_sets="dev${data_affix} eval${data_affix}"

# Get dev and eval set names from the test_sets
dev_set=$( echo $test_sets | cut -d " " -f1 )
eval_set=$( echo $test_sets | cut -d " " -f2 )

# please change the path accordingly. We need the original LibriCSS
# corpus to get the oracle segments (for evaluation purpose), and
# also the path to the separated wav files
libricss_corpus=/export/fs01/LibriCSS/

# Separated wav files
wav_files_dir=/export/c03/zhuc/css/connected_continuous_separation

##########################################################################
# We first prepare the CSS data in the Kaldi data format. We use session 0 
# for dev and others for eval. Since separation has been performed before-
# hand, each recording will contain multiple streams. We do not make any
# assumptions on the number of streams, so that this recipe is extensible
# to other speech separation methods. However, the following script may
# need to be modified depending on the naming conventions used for the
# wav files.
##########################################################################
if [ $stage -le 0 ]; then
  local/data_prep_css.sh --data-affix "$data_affix" \
    $libricss_corpus $wav_files_dir
fi

#######################################################################
# Perform SAD on the dev/eval data using py-webrtcvad package
#######################################################################

if [ $stage -le 1 ]; then
  for datadir in ${test_sets}; do
    test_set=data/${datadir}
    if [ $sad_type == "webrtc" ]; then
      echo "Applying WebRTC-VAD on ${datadir}"
      local/segmentation/apply_webrtcvad.py --mode 2 $test_set | sort > $test_set/segments
    else
      echo "Applying TDNN-Stats-SAD on ${datadir}"
      if [ ! -f ${test_set}/wav.scp ]; then
        echo "$0: Not performing SAD on ${test_set}"
        exit 0
      fi

      sad_nj=$(wc -l < "$test_set/wav.scp")
      nj=$(echo $((decode_nj>sad_nj ? sad_nj : decode_nj)))
      # Perform segmentation. We use the pretrained SAD available at:
      # http://kaldi-asr.org/models/4/0004_tdnn_stats_asr_sad_1a.tar.gz
      # Download and extract using tar -xvzf
      if [ ! -d exp/segmentation_1a/tdnn_stats_asr_sad_1a ]; then
        wget http://kaldi-asr.org/models/4/0004_tdnn_stats_asr_sad_1a.tar.gz
        tar -xvzf 0004_tdnn_stats_asr_sad_1a.tar.gz
      fi
      local/detect_speech_activity.sh --cmd "$decode_cmd" --nj $nj $test_set \
        exp/segmentation_1a/tdnn_stats_asr_sad_1a
      
      # The pretrained SAD used a different MFCC config. We need to
      # copy back our old config files.
      cp -r ../s5_mono/conf .
    fi

    # Create dummy utt2spk file from obtained segments
    awk '{print $1, $2}' ${test_set}/segments > ${test_set}/utt2spk
    utils/utt2spk_to_spk2utt.pl ${test_set}/utt2spk > ${test_set}/spk2utt
    
    # Generate RTTM file from segmentation performed by SAD. This can
    # be used to evaluate the performance of the SAD as an intermediate
    # step. Note that we remove the "stream" from the segments file reco_id
    # here because our ground truth does not have these. This means that
    # we will have overlapping segments, but that is allowed in the evaluation.
    awk '{$2=$2;sub(/_[0-9]*$/, "", $2); print}' ${test_set}/segments > ${test_set}/segments.score
    steps/segmentation/convert_utt2spk_and_segments_to_rttm.py \
      ${test_set}/utt2spk ${test_set}/segments.score ${test_set}/rttm
    rm $test_set/segments.score

    echo "Scoring $datadir.."
    # We first generate the reference RTTM from the backed up utt2spk and segments
    # files.
    ref_rttm=${test_set}/ref_rttm
    steps/segmentation/convert_utt2spk_and_segments_to_rttm.py ${test_set}/utt2spk.bak \
      ${test_set}/segments.bak ${test_set}/ref_rttm

    md-eval.pl -r $ref_rttm -s ${test_set}/rttm |\
      awk 'or(/MISSED SPEECH/,/FALARM SPEECH/)'
    
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
    utils/fix_data_dir.sh data/$x
    nj=$(wc -l < "data/$x/wav.scp")
    steps/make_mfcc.sh --nj $decode_nj --cmd "$train_cmd" \
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

    [ ! -d exp/xvector_nnet_1a ] && ./local/download_diarizer.sh

    nj=$(echo $((decode_nj>diar_nj ? diar_nj : decode_nj)))
    local/diarize_css.sh --nj $nj --cmd "$train_cmd" --stage $diarizer_stage \
      --ref-rttm $ref_rttm --post-process-rttm true \
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
    nj=$(echo $((decode_nj>asr_nj ? asr_nj : decode_nj)))
    local/decode_diarized_css.sh --nj $nj --cmd "$decode_cmd" --stage $decode_diarize_stage \
      --lm-suffix "_tgsmall" --acwt 1.0 --post-decode-acwt 10.0 \
      exp/${datadir}_diarization/rttm.post data/$datadir data/lang_test_tgsmall \
      exp/chain${nnet3_affix}/tdnn_${affix} exp/nnet3${nnet3_affix} \
      data/${datadir}_diarized || exit 1
  done
fi

#######################################################################
# Score decoded dev/eval sets (only if we are not rescoring)
#######################################################################
if [ $stage -le 5 ] && [ ! $rnnlm_rescore ]; then
  # please specify both dev and eval set directories so that the search parameters
  # (insertion penalty and language model weight) will be tuned using the dev set
  local/score_reco_diarized.sh --cmd "$train_cmd" --stage $score_stage \
      --multistream true \
      --dev_decodedir exp/chain${nnet3_affix}/tdnn_${affix}/decode_${dev_set}_diarized_2stage \
      --dev_datadir ${dev_set}_diarized_hires \
      --eval_decodedir exp/chain${nnet3_affix}/tdnn_${affix}/decode_${eval_set}_diarized_2stage \
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
      rnnlm/lmrescore$pruned.sh \
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
        --multistream true \
        --dev_decodedir exp/chain${nnet3_affix}/tdnn_${affix}/decode_${dev_set}_diarized_2stage_rescore \
        --dev_datadir ${dev_set}_diarized_hires \
        --eval_decodedir exp/chain${nnet3_affix}/tdnn_${affix}/decode_${eval_set}_diarized_2stage_rescore \
        --eval_datadir ${eval_set}_diarized_hires
  fi
fi

exit 0;

