#!/usr/bin/env bash

# Copyright 2016-17  Vimal Manohar
#              2017  Nagendra Kumar Goel
# Copyright  2020 John Morgan (ARL)
# Apache 2.0.

# This script does nnet3-based speech activity detection.

set -e 
set -o pipefail
set -u

if [ -f ./path.sh ]; then . ./path.sh; fi

affix=1a  # Affix for the segmentation
nj=10
cmd=run.pl
stage=-1

# Feature options (Must match training)
mfcc_config=conf/mfcc_hires.conf
output_name=output   # The output node in the network

# SAD network config
iter=final  # Model iteration to use

# Contexts must ideally match training for LSTM models
extra_left_context=60  # Set to some large value, typically 40 for LSTM (must match training)
extra_right_context=10  
extra_left_context_initial=0
extra_right_context_final=0
frames_per_chunk=150

# Decoding options
graph_opts="--min-silence-duration=0.03 --min-speech-duration=0.3 --max-speech-duration=10.0"
acwt=0.3

# These <from>_in_<to>_weight represent the fraction of <from> probability 
# to transfer to <to> class.
# e.g. --speech-in-sil-weight=0.0 --garbage-in-sil-weight=0.0 --sil-in-speech-weight=0.0 --garbage-in-speech-weight=0.3
transform_probs_opts=""

# Postprocessing options
segment_padding=0.2   # Duration (in seconds) of padding added to segments 
# min_segment_dur:
# Minimum duration (in seconds) required for a segment to be included
# This is before any padding. Segments shorter than this duration will be removed.
# This is an alternative to --min-speech-duration above.
min_segment_dur=0
# merge_consecutive_max_dur:
#Merge consecutive segments
# as long as the merged segment is no longer than this many seconds.
#The segments are only merged if their boundaries are touching.
# This is after padding by --segment-padding seconds.
# 0 means do not merge. Use 'inf' to not limit the duration.
merge_consecutive_max_dur=inf

echo "$0 $*"

. utils/parse_options.sh

if [ $# -ne 1 ]; then
  echo "This script does nnet3-based speech activity detection."
  echo "Input is a fold corresponding to a kaldi  directory."
  echo "Output is also a kaldi data directory."
  echo "Usage: $0 <FOLD>"
  echo "<FOLD>: The data fold."
  echo "The directory data/<FOLD> must exist."
  echo "For example :"
  echo "$0 dev-1"
  exit 1
fi

fld=$1

dir=exp/sad_${affix}
model_dir=$dir/tdnn_lstm_sad_${affix}

if [ $stage -le 0 ]; then
    echo "$0 Stage 0: Convert $fld directory to whole."
    utils/data/convert_data_dir_to_whole.sh data/$fld data/${fld}_whole
fi

if [ $stage -le 1 ]; then
  echo "$0 Stage 1: Extract input features in data/${fld}_whole."
  utils/fix_data_dir.sh data/${fld}_whole
  steps/make_mfcc.sh --mfcc-config $mfcc_config --nj $nj --cmd "$cmd" \
    --write-utt2num-frames true \
    data/${fld}_whole 
  steps/compute_cmvn_stats.sh data/${fld}_whole
  utils/fix_data_dir.sh data/${fld}_whole
fi

frame_subsampling_factor=1
mkdir -p $dir/$fld
if [ $stage -le 2 ]; then
  echo "$0 Stage 2: Forward pass through the network and dump log-likelihoods for $fld."
  steps/nnet3/compute_output.sh --nj $nj --cmd "$cmd" \
    --iter ${iter} \
    --extra-left-context $extra_left_context \
    --extra-right-context $extra_right_context \
    --extra-left-context-initial $extra_left_context_initial \
    --extra-right-context-final $extra_right_context_final \
    --frames-per-chunk $frames_per_chunk --apply-exp true \
    --frame-subsampling-factor $frame_subsampling_factor \
    data/${fld}_whole $model_dir $dir/$fld || exit 1
fi

utils/data/get_utt2dur.sh --nj $nj --cmd "$cmd" data/${fld}_whole || exit 1
frame_shift=$(utils/data/get_frame_shift.sh data/${fld}_whole) || exit 1
graph_dir=${dir}/graph_${output_name}

if [ $stage -le 3 ]; then
  echo "$0 Stage 3:  Prepare FST to make speech/silence decisions."
  mkdir -p $graph_dir
  # 1 for silence and 2 for speech
  cat <<EOF > $graph_dir/words.txt
<eps> 0
silence 1
speech 2
EOF

  $cmd $graph_dir/log/make_graph.log \
    steps/segmentation/internal/prepare_sad_graph.py $graph_opts \
      --frame-shift=$(perl -e "print $frame_shift * $frame_subsampling_factor") - \| \
    fstcompile --isymbols=$graph_dir/words.txt --osymbols=$graph_dir/words.txt '>' \
      $graph_dir/HCLG.fst
fi

post_vec=$model_dir/post_${output_name}.vec
if [ $stage -le 4 ]; then
  echo "$0 Stage 4: Getting probability matrix."
  local/get_transform_probs_mat.py \
    --priors="$post_vec" $transform_probs_opts > $dir/$fld/transform_probs.mat
fi

if [ $stage -le 5 ]; then
  echo "$0 Stage 5: Decoding."
  steps/segmentation/decode_sad.sh --acwt $acwt --cmd "$cmd" \
    --nj $nj \
    --transform "$dir/$fld/transform_probs.mat" \
    $graph_dir $dir/$fld $dir/$fld
fi

if [ $stage -le 6 ]; then
    echo "$0 Stage 6: Post-process segmentation to create kaldi data directory."
  steps/segmentation/post_process_sad_to_segments.sh \
    --segment-padding $segment_padding --min-segment-dur $min_segment_dur \
    --merge-consecutive-max-dur $merge_consecutive_max_dur \
    --cmd "$cmd" --frame-shift $(perl -e "print $frame_subsampling_factor * $frame_shift") \
    data/${fld}_whole $dir/$fld $dir/$fld
fi

if [ $stage -le 7 ]; then
  echo "$0 Stage 7: Subsegmenting."
  utils/data/subsegment_data_dir.sh data/${fld}_whole $dir/$fld/segments \
    data/${fld}_seg
  cp data/$fld/wav.scp data/${fld}_seg
  cp data/$fld/{stm,reco2file_and_channel,glm} data/${fld}_seg/ || true
  utils/fix_data_dir.sh data/${fld}_seg
fi

echo "$0: Created output segmented kaldi data directory in data/${fld}_seg."
exit 0
