#!/bin/bash

set -e 
set -o pipefail
set -u

. path.sh
. cmd.sh

affix=  # Affix for the segmentation
reco_nj=32  # works on recordings as against on speakers

# Feature options (Must match training)
mfcc_config=conf/mfcc_hires_bp.conf
feat_affix=bp   # Affix for the type of feature used

stage=-1
sad_stage=-1
output_name=output-speech   # The output node in the network
sad_name=sad    # Base name for the directory storing the computed loglikes
segmentation_name=segmentation  # Base name for the directory doing segmentation

# SAD network config
iter=final  # Model iteration to use

# Contexts must ideally match training for LSTM models, but
# may not necessarily for stats components
extra_left_context=0  # Set to some large value, typically 40 for LSTM (must match training)
extra_right_context=0  

frame_subsampling_factor=3  # Subsampling at the output

# Set to true if the test data has > 8kHz sampling frequency.
do_downsampling=false

# Segmentation configs
min_silence_duration=30
min_speech_duration=30
sil_prior=0.5
speech_prior=0.5
segmentation_config=conf/segmentation_speech.conf
convert_data_dir_to_whole=true

echo $* 

. utils/parse_options.sh

if [ $# -ne 4 ]; then
  echo "Usage: $0 <src-data-dir> <sad-nnet-dir> <mfcc-dir> <data-dir>"
  echo " e.g.: $0 ~/workspace/egs/ami/s5b/data/sdm1/dev exp/nnet3_sad_snr/nnet_tdnn_j_n4 mfcc_hires_bp data/ami_sdm1_dev"
  exit 1
fi

src_data_dir=$1   # The input data directory that needs to be segmented.
                  # Any segments in that will be ignored.
sad_nnet_dir=$2   # The SAD neural network
mfcc_dir=$3       # The directory to store the features
data_dir=$4       # The output data directory will be ${data_dir}_seg

affix=${affix:+_$affix}
feat_affix=${feat_affix:+_$feat_affix}

data_id=`basename $data_dir`
sad_dir=${sad_nnet_dir}/${sad_name}${affix}_${data_id}_whole${feat_affix}
seg_dir=${sad_nnet_dir}/${segmentation_name}${affix}_${data_id}_whole${feat_affix}

export PATH="$KALDI_ROOT/tools/sph2pipe_v2.5/:$PATH"
[ ! -z `which sph2pipe` ]

whole_data_dir=${sad_dir}/${data_id}_whole

if $convert_data_dir_to_whole; then
  if [ $stage -le 0 ]; then
    utils/data/convert_data_dir_to_whole.sh $src_data_dir ${whole_data_dir}
    
    if $do_downsampling; then
      freq=`cat $mfcc_config | perl -pe 's/\s*#.*//g' | grep "sample-frequency=" | awk -F'=' '{if (NF == 0) print 16000; else print $2}'`
      utils/data/downsample_data_dir.sh $freq $whole_data_dir
    fi

    utils/copy_data_dir.sh ${whole_data_dir} ${whole_data_dir}${feat_affix}_hires
  fi

  if [ $stage -le 1 ]; then
    steps/make_mfcc.sh --mfcc-config $mfcc_config --nj $reco_nj --cmd "$train_cmd" \
      ${whole_data_dir}${feat_affix}_hires exp/make_hires/${data_id}_whole${feat_affix} $mfcc_dir
    steps/compute_cmvn_stats.sh ${whole_data_dir}${feat_affix}_hires exp/make_hires/${data_id}_whole${feat_affix} $mfcc_dir
    utils/fix_data_dir.sh ${whole_data_dir}${feat_affix}_hires
  fi
  test_data_dir=${whole_data_dir}${feat_affix}_hires
else
  test_data_dir=$src_data_dir
fi

post_vec=$sad_nnet_dir/post_${output_name}.vec
if [ ! -f $sad_nnet_dir/post_${output_name}.vec ]; then
  echo "$0: Could not find $sad_nnet_dir/post_${output_name}.vec. See the last stage of local/segmentation/run_train_sad.sh"
  exit 1
fi

if [ $stage -le 2 ]; then
  steps/nnet3/compute_output.sh --nj $reco_nj --cmd "$train_cmd" \
    --post-vec "$post_vec" \
    --iter $iter \
    --extra-left-context $extra_left_context \
    --extra-right-context $extra_right_context \
    --frames-per-chunk 150 \
    --stage $sad_stage --output-name $output_name \
    --frame-subsampling-factor $frame_subsampling_factor \
    --use-raw-nnet true ${test_data_dir} $sad_nnet_dir $sad_dir
fi

if [ $stage -le 3 ]; then
  steps/segmentation/decode_sad_to_segments.sh \
    --use-unigram-lm false \
    --frame-subsampling-factor $frame_subsampling_factor \
    --min-silence-duration $min_silence_duration \
    --min-speech-duration $min_speech_duration \
    --sil-prior $sil_prior \
    --speech-prior $speech_prior \
    --segmentation-config $segmentation_config --cmd "$train_cmd" \
    ${test_data_dir} $sad_dir $seg_dir ${data_dir}_seg
fi

# Subsegment data directory
if [ $stage -le 4 ]; then
  rm ${data_dir}_seg/feats.scp || true
  utils/data/get_reco2num_frames.sh --cmd "$train_cmd" --nj $reco_nj ${test_data_dir} 
  awk '{print $1" "$2}' ${data_dir}_seg/segments | \
    utils/apply_map.pl -f 2 ${test_data_dir}/reco2num_frames > \
    ${data_dir}_seg/utt2max_frames

  frame_shift_info=`cat $mfcc_config | steps/segmentation/get_frame_shift_info_from_config.pl`
  utils/data/get_subsegment_feats.sh ${test_data_dir}/feats.scp \
    $frame_shift_info ${data_dir}_seg/segments | \
    utils/data/fix_subsegmented_feats.pl ${data_dir}_seg/utt2max_frames > \
    ${data_dir}_seg/feats.scp
  steps/compute_cmvn_stats.sh --fake ${data_dir}_seg

  utils/fix_data_dir.sh ${data_dir}_seg
fi
