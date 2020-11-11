#!/usr/bin/env bash

# Copyright 2016-17  Vimal Manohar
#              2017  Nagendra Kumar Goel
# Apache 2.0.

# This script does nnet3-based speech activity detection given an input 
# kaldi data directory and outputs a segmented kaldi data directory.

set -e 
set -o pipefail
set -u

if [ -f ./path.sh ]; then . ./path.sh; fi

affix=  # Affix for the segmentation
nj=32
cmd=queue.pl
stage=-1

# Feature options (Must match training)
mfcc_config=conf/mfcc_hires.conf
feat_affix=   # Affix for the type of feature used

output_name=output   # The output node in the network
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
extra_left_context_initial=-1
extra_right_context_final=-1
frames_per_chunk=150

# Decoding options
graph_opts="--min-silence-duration=0.03 --min-speech-duration=0.3 --max-speech-duration=10.0"
acwt=1.0

# These <from>_in_<to>_weight represent the fraction of <from> probability 
# to transfer to <to> class.
# e.g. --speech-in-sil-weight=0.0 --garbage-in-sil-weight=0.0 --sil-in-speech-weight=0.0 --garbage-in-speech-weight=0.3
transform_probs_opts=""

# Postprocessing options
segment_padding=0.2   # Duration (in seconds) of padding added to segments 
min_segment_dur=0   # Minimum duration (in seconds) required for a segment to be included
                    # This is before any padding. Segments shorter than this duration will be removed.
                    # This is an alternative to --min-speech-duration above.
merge_consecutive_max_dur=0   # Merge consecutive segments as long as the merged segment is no longer than this many
                              # seconds. The segments are only merged if their boundaries are touching.
                              # This is after padding by --segment-padding seconds.
                              # 0 means do not merge. Use 'inf' to not limit the duration.
cleanup=false  # If true, remove files created during feature extraction

echo $* 

. utils/parse_options.sh

if [ $# -ne 2 ]; then
  echo "This script does nnet3-based speech activity detection given an input kaldi "
  echo "data directory and outputs an output kaldi data directory."
  echo "See script for details of the options to be supplied."
  echo "Usage: $0 <src-data-dir> <sad-nnet-dir>"
  echo " e.g.: $0 ~/workspace/egs/ami/s5b/data/sdm1/dev exp/nnet3_sad_snr/nnet_tdnn_j_n4 \\"
  echo "    mfcc_hires exp/segmentation_sad_snr/nnet_tdnn_j_n4"
  echo ""
  echo "Options: "
  echo "  --cmd (utils/run.pl|utils/queue.pl <queue opts>) # how to run jobs."
  echo "  --nj <num-job>                                 # number of parallel jobs to run."
  echo "  --stage <stage>                                # stage to do partial re-run from."
  echo "  --convert-data-dir-to-whole <true|false>    # If true, the input data directory is "
  echo "                                              # first converted to whole data directory (i.e. whole recordings) "
  echo "                                              # and segmentation is done on that."
  echo "                                              # If false, then the original segments are "
  echo "                                              # retained and they are split into sub-segments."
  echo "  --output-name <name>    # The output node in the network"
  echo "  --extra-left-context  <context|0>   # Set to some large value, typically 40 for LSTM (must match training)"
  echo "  --extra-right-context  <context|0>   # For BLSTM or statistics pooling"
  echo "  --cleanup <true|false>  # Remove files created during feature extraction"
  exit 1
fi

src_data_dir=$1   # The input data directory that needs to be segmented.
                  # If convert_data_dir_to_whole is true, any segments in that will be ignored.
sad_nnet_dir=$2   # The SAD neural network

dir=exp/segmentation${affix}

affix=${affix:+_$affix}
feat_affix=${feat_affix:+_$feat_affix}

data_id=`basename $src_data_dir`
sad_dir=${dir}/${sad_name}${affix}_${data_id}${feat_affix}
seg_dir=${dir}/${segmentation_name}${affix}_${data_id}${feat_affix}

###############################################################################
## Forward pass through the network network and dump the log-likelihoods.
###############################################################################

frame_subsampling_factor=1
if [ -f $sad_nnet_dir/frame_subsampling_factor ]; then
  frame_subsampling_factor=$(cat $sad_nnet_dir/frame_subsampling_factor)
fi

if [ $stage -le 1 ]; then
  # mfccdir should be some place with a largish disk where you
  # want to store MFCC features.
  mfccdir=mfcc
  steps/make_mfcc.sh --nj $nj --cmd "$train_cmd" \
    --mfcc-config conf/mfcc_hires.conf \
    $src_data_dir exp/make_mfcc/$data_id $mfccdir
fi

mkdir -p $dir
if [ $stage -le 2 ]; then
  if [ "$(readlink -f $sad_nnet_dir)" != "$(readlink -f $dir)" ]; then
    cp $sad_nnet_dir/cmvn_opts $dir || exit 1
  fi

  ########################################################################
  ## Initialize neural network for decoding using the output $output_name
  ########################################################################

  if [ ! -z "$output_name" ] && [ "$output_name" != output ]; then
    $cmd $dir/log/get_nnet_${output_name}.log \
      nnet3-copy --edits="rename-node old-name=$output_name new-name=output" \
      $sad_nnet_dir/$iter.raw $dir/${iter}_${output_name}.raw || exit 1
    iter=${iter}_${output_name}
  else 
    if ! diff $sad_nnet_dir/$iter.raw $dir/$iter.raw; then
      cp $sad_nnet_dir/$iter.raw $dir/
    fi
  fi

  steps/nnet3/compute_output.sh --nj $nj --cmd "$cmd" \
    --iter ${iter} \
    --extra-left-context $extra_left_context \
    --extra-right-context $extra_right_context \
    --extra-left-context-initial $extra_left_context_initial \
    --extra-right-context-final $extra_right_context_final \
    --frames-per-chunk $frames_per_chunk --apply-exp true \
    --frame-subsampling-factor $frame_subsampling_factor \
    ${src_data_dir} $dir $sad_dir || exit 1
fi

###############################################################################
## Prepare FST we search to make speech/silence decisions.
###############################################################################

utils/data/get_utt2dur.sh --nj $nj --cmd "$cmd" $src_data_dir || exit 1
frame_shift=$(utils/data/get_frame_shift.sh $src_data_dir) || exit 1

graph_dir=${dir}/graph_${output_name}
if [ $stage -le 3 ]; then
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

###############################################################################
## Do Viterbi decoding to create per-frame alignments.
###############################################################################

post_vec=$sad_nnet_dir/post_${output_name}.vec
if [ ! -f $sad_nnet_dir/post_${output_name}.vec ]; then
  if [ ! -f $sad_nnet_dir/post_${output_name}.txt ]; then
    echo "$0: Could not find $sad_nnet_dir/post_${output_name}.vec. "
    echo "Re-run the corresponding stage in the training script possibly "
    echo "with --compute-average-posteriors=true or compute the priors "
    echo "from the training labels"
    exit 1
  else
    post_vec=$sad_nnet_dir/post_${output_name}.txt
  fi
fi

mkdir -p $seg_dir
if [ $stage -le 4 ]; then
  steps/segmentation/internal/get_transform_probs_mat.py \
    --priors="$post_vec" $transform_probs_opts > $seg_dir/transform_probs.mat

  steps/segmentation/decode_sad.sh --acwt $acwt --cmd "$cmd" \
    --nj $nj \
    --transform "$seg_dir/transform_probs.mat" \
    $graph_dir $sad_dir $seg_dir
fi

###############################################################################
## Post-process segmentation to create kaldi data directory.
###############################################################################

if [ $stage -le 5 ]; then
  steps/segmentation/post_process_sad_to_segments.sh \
    --segment-padding $segment_padding --min-segment-dur $min_segment_dur \
    --merge-consecutive-max-dur $merge_consecutive_max_dur \
    --cmd "$cmd" --frame-shift $(perl -e "print $frame_subsampling_factor * $frame_shift") \
    ${src_data_dir} ${seg_dir} ${seg_dir}
fi

sed 's:-:_:g' ${seg_dir}/segments > $src_data_dir/segments # to be consistent for scoring

if [ $cleanup ]; then
  rm $src_data_dir/{feats.scp,frame_shift,utt2dur,utt2num_frames} 2> /dev/null
fi

echo "$0: Created output segments in ${src_data_dir}"
exit 0