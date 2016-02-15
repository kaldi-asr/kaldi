#!/bin/bash

set -e 
set -o pipefail
set -u

. path.sh
. cmd.sh

feat_affix=_bp_vh
snr_affix=
affix=_babel_assamese_r2000_a
reco_nj=32

src_data_dir=data/dev10h.pem
data_dir=data/babel_assamese_dev10h
irm_predictor=exp/nnet3_irm_predictor/nnet_tdnn_a_w_bp_vh_seg_babel_assamese_unsad_r2000_n6_lrate0.00003_0.000003
predictor_iter=final
sad_nnet_dir=exp/nnet3_sad_snr/tdnn_irm_babel_assamese_train_unsad_splice5_2
sad_nnet_iter=final
append_to_orig_feats=false
create_uniform_segments=false
overlap_length=500
window_length=3000
stage=-1
feature_type=Snr

segmentation_config=conf/segmentation.conf
weights_segmentation_config=conf/segmentation.conf

echo $* 

. utils/parse_options.sh

if [ $# -ne 4 ]; then
  echo "Usage: $0 <src-data-dir> <data-dir> <irm-predictor> <sad-nnet-dir>"
  echo " e.g.: $0 $src_data_dir $data_dir $irm_predictor $sad_nnet_dir"
  exit 1
fi

src_data_dir=$1
data_dir=$2
irm_predictor=$3
sad_nnet_dir=$4

data_id=`basename $data_dir`
snr_data_dir=exp/frame_snrs_irm${affix}${feat_affix}_${data_id}_whole${feat_affix}/${data_id}${feat_affix}_snr${snr_affix}
sad_dir=${sad_nnet_dir}/sad${affix}_${data_id}_whole${feat_affix}
seg_dir=${sad_nnet_dir}/segmentation${affix}_${data_id}_whole${feat_affix}

if [ $stage -le 0 ]; then
  diarization/convert_data_dir_to_whole.sh $src_data_dir ${data_dir}_whole
  utils/copy_data_dir.sh ${data_dir}_whole ${data_dir}_whole${feat_affix}_hires
  utils/copy_data_dir.sh ${data_dir}_whole ${data_dir}_whole${feat_affix}_fbank
fi

if [ $stage -le 1 ]; then
  steps/make_mfcc.sh --mfcc-config conf/mfcc_hires_bp.conf --nj $reco_nj --cmd "$train_cmd" \
    ${data_dir}_whole${feat_affix}_hires exp/make_hires/${data_id}_whole${feat_affix} mfcc_hires
  steps/compute_cmvn_stats.sh ${data_dir}_whole${feat_affix}_hires exp/make_hires/${data_id}_whole${feat_affix} mfcc_hires
fi

if [ $stage -le 2 ]; then
  steps/make_fbank.sh --fbank-config conf/fbank_bp.conf --nj $reco_nj --cmd "$train_cmd" \
    ${data_dir}_whole${feat_affix}_fbank exp/make_fbank/${data_id}_whole${feat_affix} fbank
  steps/compute_cmvn_stats.sh ${data_dir}_whole${feat_affix}_fbank exp/make_fbank/${data_id}_whole${feat_affix} fbank
fi

if [ $stage -le 3 ]; then
  local/snr/compute_frame_snrs.sh --use-gpu no --nj 32 \
    --cmd "$decode_cmd --max-jobs-run 32" --iter $predictor_iter \
    $irm_predictor ${data_dir}_whole${feat_affix}_hires ${data_dir}_whole${feat_affix}_fbank \
    exp/frame_snrs_irm${affix}${feat_affix}_${data_id}_whole${feat_affix}
fi

if [ $stage -le 4 ]; then
  local/snr/create_snr_data_dir.sh --cmd "$train_cmd" --nj $reco_nj --type $feature_type \
    --add-frame-snr true --append-to-orig-feats $append_to_orig_feats --dataid ${data_id}_whole${feat_affix} \
    ${data_dir}_whole${feat_affix}_fbank exp/frame_snrs_irm${affix}${feat_affix}_${data_id}_whole${feat_affix} \
    exp/make_snr_data_dir snr_feats $snr_data_dir
fi

if [ $stage -le 5 ]; then
  local/snr/compute_sad.sh --snr-data-dir $snr_data_dir --use-gpu no --method Dnn \
    --iter $sad_nnet_iter $sad_nnet_dir $snr_data_dir $sad_dir
fi

if ! $create_uniform_segments; then
  if [ $stage -le 6 ]; then
    local/snr/sad_to_segments.sh --config $segmentation_config --acwt 0.1 \
      --speech-to-sil-ratio 1.0 --sil-self-loop-probability 0.5 \
      --sil-transition-probability 0.5 ${data_dir}_whole${feat_affix}_fbank \
      $sad_dir $seg_dir $seg_dir/${data_id}_seg
  fi

  if [ $stage -le 7 ]; then
    local/snr/get_weights_for_ivector_extraction.sh --cmd queue.pl \
      --method Viterbi --config $weights_segmentation_config \
      --silence-weight 0 \
      ${seg_dir}/${data_id}_seg ${sad_dir} $seg_dir/ivector_weights_${data_id}_seg
  fi
else
  if [ $stage -le 6 ]; then
    local/snr/uniform_segment_data_dir.sh --overlap-length $overlap_length \
      --window-length $window_length \
      ${data_dir}_whole${feat_affix}_fbank $seg_dir $seg_dir/${data_id}_uniform_seg
  fi
  if [ $stage -le 7 ]; then
    local/snr/get_weights_for_ivector_extraction.sh --cmd queue.pl \
      --method Viterbi --config $weights_segmentation_config \
      --silence-weight 0 \
      ${seg_dir}/${data_id}_uniform_seg ${sad_dir} $seg_dir/ivector_weights_${data_id}_uniform_seg
  fi
fi
