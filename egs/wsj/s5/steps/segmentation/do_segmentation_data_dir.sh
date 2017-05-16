#! /bin/bash

# Copyright 2016  Vimal Manohar
# Apache 2.0.

# This script is deprecated in favor of do_segmentation_data_dir_simple.sh.
# This script does nnet3-based speech activity detection given an input kaldi
# directory and outputs an output kaldi directory.
# This script can also do music detection using appropriate options 
# such as --output-name output-music.

set -e 
set -o pipefail
set -u

. path.sh
. cmd.sh

affix=      # Affix for the segmentation
nj=4
stage=-1
sad_stage=-1

# Feature options (Must match feats used to train nnet3 model)
# Applicable only when convert_data_dir_to_whole is true
convert_data_dir_to_whole=true
mfcc_config=conf/mfcc_hires_bp.conf
feat_affix=bp   # Affix for the type of feature used

output_name=output-speech   # The output node in the network. 
sad_name=sad    # Base name for the directory storing the computed loglikes
                # Can be music for music detection
segmentation_name=segmentation  # Base name for the directory doing segmentation
                                # Can be segmentation_music for music detection

# SAD network config
iter=final  # Model iteration to use

# Contexts must ideally match training for LSTM models, but
# may not necessarily for stats components
extra_left_context=0  # Set to some large value, typically 40 for LSTM (must match training)
extra_right_context=0  

frame_subsampling_factor=3    # Subsampling at the output

# Set to true if the test data has > 8kHz sampling frequency and 
# requires downsampling
do_downsampling=false

# Segmentation config
min_silence_duration=30   # Minimum duration of silence (in 10ms units)
min_speech_duration=30    # Minimum duratoin of speech (in 10ms units)
sil_prior=0.5       # Prior probability on silence
speech_prior=0.5    # Prior probability on speech

# Post-processing options passed to post_process_sad_to_segments.sh
segmentation_config=conf/segmentation_speech.conf   

echo $* 

. utils/parse_options.sh

if [ $# -ne 4 ]; then
  echo "Usage: $0 <src-data-dir> <sad-nnet-dir> <mfcc-dir> <data-dir>"
  echo " e.g.: $0 ~/workspace/egs/ami/s5b/data/sdm1/dev exp/nnet3_sad_snr/nnet_tdnn_j_n4 mfcc_hires_bp data/ami_sdm1_dev"
  exit 1
fi

src_data_dir=$1   # The input data directory that needs to be segmented.
                  # If convert_data_dir_to_whole is true, any segments in that will be ignored.
sad_nnet_dir=$2   # The SAD neural network
mfcc_dir=$3       # The directory to store the features
data_dir=$4       # The output data directory will be ${data_dir}_seg

affix=${affix:+_$affix}
feat_affix=${feat_affix:+_$feat_affix}

data_id=`basename $data_dir`
sad_dir=${sad_nnet_dir}/${sad_name}${affix}_${data_id}_whole${feat_affix}
seg_dir=${sad_nnet_dir}/${segmentation_name}${affix}_${data_id}_whole${feat_affix}

if $convert_data_dir_to_whole; then
  whole_data_dir=${sad_dir}/${data_id}_whole

  if [ $stage -le 0 ]; then
    utils/data/convert_data_dir_to_whole.sh $src_data_dir ${whole_data_dir}
    
    if $do_downsampling; then
      freq=`cat $mfcc_config | perl -pe 's/\s*#.*//g' | grep "sample-frequency=" | awk -F'=' '{if (NF == 0) print 16000; else print $2}'`
      utils/data/downsample_data_dir.sh $freq $whole_data_dir
    fi

    utils/copy_data_dir.sh ${whole_data_dir} ${whole_data_dir}${feat_affix}_hires
  fi

  if [ $stage -le 1 ]; then
    steps/make_mfcc.sh --mfcc-config $mfcc_config --nj $nj --cmd "$train_cmd" \
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
  echo "$0: Could not find $sad_nnet_dir/post_${output_name}.vec."
  echo "Re-run the corresponding stage in the training script."
  echo "See local/segmentation/train_lstm_sad_music.sh for example."
  exit 1
fi

if [ $stage -le 2 ]; then
  steps/nnet3/compute_output.sh --nj $nj --cmd "$train_cmd" \
    --priors "$post_vec" \
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
